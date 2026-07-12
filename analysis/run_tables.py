from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "results"

OFFLINE_METHODS = [
    "top_score",
    "random_candidate",
    "vlm_direct",
    "first_question",
    "random_question",
    "vlm_best_question",
    "proposed_efe",
]
ONLINE_METHODS = ["top_score", "random_candidate", "vlm_best_question", "proposed_efe"]
INTERACTIVE_METHODS = ["first_question", "random_question", "vlm_best_question", "proposed_efe"]
OFFLINE_BOOTSTRAP_SEED = 20260520
ONLINE_RANDOM_SEED = 20260605
OFFLINE_BOOTSTRAP_ITERATIONS = 10_000
ONLINE_BOOTSTRAP_ITERATIONS = 2_000


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return float(center - margin), float(center + margin)


def holm(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite = np.where(np.isfinite(values))[0]
    if not len(finite):
        return adjusted.tolist()
    order = finite[np.argsort(values[finite])]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * values[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def paired_bootstrap_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    iterations: int = OFFLINE_BOOTSTRAP_ITERATIONS,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    draws = values[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def offline_table5(offline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in OFFLINE_METHODS:
        group = offline[offline["method"] == method]
        n = len(group)
        successes = int(group["target_correct"].sum())
        asked = int(group["asked"].sum())
        low, high = wilson(successes, n)
        asked_questions = group.loc[group["asked"], "question_count"]
        rows.append(
            {
                "method": method,
                "n": n,
                "correct": successes,
                "target_accuracy_percent": 100.0 * successes / n,
                "wilson_ci_low_percent": 100.0 * low,
                "wilson_ci_high_percent": 100.0 * high,
                "asked_percent": 100.0 * asked / n,
                "mean_questions_all_trials": float(group["question_count"].mean()),
                "mean_questions_asked_trials": float(asked_questions.mean()) if len(asked_questions) else 0.0,
                "mean_latency_s": float(group["latency"].mean()),
                "median_latency_s": float(group["latency"].median()),
            }
        )
    return pd.DataFrame(rows)


def offline_table6(offline: pd.DataFrame) -> pd.DataFrame:
    work = offline.copy()
    work["target_correct_int"] = work["target_correct"].astype(int)
    result = smf.gee(
        "target_correct_int ~ C(method, Treatment(reference='proposed_efe'))"
        " + C(instruction_type) + C(scene_family) + candidate_count",
        groups="scene_id",
        data=work,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()
    params = result.params
    covariance = result.cov_params()

    def contrast(left: str, right: str) -> tuple[float, float, float]:
        vector = pd.Series(0.0, index=params.index)
        for method, sign in ((left, 1.0), (right, -1.0)):
            key = f"C(method, Treatment(reference='proposed_efe'))[T.{method}]"
            if key in vector.index:
                vector[key] += sign
        log_odds = float(vector @ params)
        standard_error = float(np.sqrt(vector @ covariance @ vector))
        p_value = float(2 * stats.norm.sf(abs(log_odds / standard_error)))
        return log_odds, standard_error, p_value

    comparisons = [
        ("proposed_efe", "top_score"),
        ("proposed_efe", "random_candidate"),
        ("proposed_efe", "vlm_direct"),
        ("vlm_best_question", "vlm_direct"),
    ]
    rows = []
    for left, right in comparisons:
        log_odds, standard_error, p_value = contrast(left, right)
        rows.append(
            {
                "comparison": f"{left} vs {right}",
                "odds_ratio": float(np.exp(log_odds)),
                "log_odds_ratio": log_odds,
                "std_error": standard_error,
                "p_value": p_value,
            }
        )
    for row, adjusted in zip(rows, holm([row["p_value"] for row in rows])):
        row["holm_p"] = adjusted
    return pd.DataFrame(rows)


def question_table7(offline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    asked = offline[offline["asked"]].copy()
    summary_rows = []
    for context, frame in [
        ("overall_asked_only", asked),
        ("ambiguous_asked_only", asked[asked["instruction_type"] == "ambiguous"]),
        ("partial_asked_only", asked[asked["instruction_type"] == "partial"]),
    ]:
        for method in INTERACTIVE_METHODS:
            values = frame.loc[frame["method"] == method, "question_count"]
            summary_rows.append(
                {
                    "context": context,
                    "method": method,
                    "n_asked_trials": int(len(values)),
                    "pooled_mean_questions": float(values.mean()) if len(values) else np.nan,
                    "pooled_median_questions": float(values.median()) if len(values) else np.nan,
                }
            )

    rng = np.random.default_rng(OFFLINE_BOOTSTRAP_SEED)
    comparison_rows = []
    for context, frame in [
        ("overall_asked_only", asked),
        ("ambiguous_asked_only", asked[asked["instruction_type"] == "ambiguous"]),
        ("partial_asked_only", asked[asked["instruction_type"] == "partial"]),
    ]:
        start = len(comparison_rows)
        efe_by_scene = (
            frame[frame["method"] == "proposed_efe"]
            .groupby("scene_id")["question_count"]
            .mean()
            .rename("efe")
        )
        for baseline in ["first_question", "random_question", "vlm_best_question"]:
            baseline_by_scene = (
                frame[frame["method"] == baseline]
                .groupby("scene_id")["question_count"]
                .mean()
                .rename("baseline")
            )
            paired = pd.concat([efe_by_scene, baseline_by_scene], axis=1).dropna()
            differences = (paired["efe"] - paired["baseline"]).to_numpy(dtype=float)
            statistic, p_value = stats.wilcoxon(differences, zero_method="wilcox")
            low, high = paired_bootstrap_ci(differences, rng)
            mean_efe = float(paired["efe"].mean())
            mean_baseline = float(paired["baseline"].mean())
            comparison_rows.append(
                {
                    "context": context,
                    "comparison": f"proposed_efe vs {baseline}",
                    "n_paired_scenes": int(len(paired)),
                    "scene_mean_questions_proposed_efe": mean_efe,
                    "scene_mean_questions_baseline": mean_baseline,
                    "mean_difference": float(differences.mean()),
                    "percent_reduction": 100.0 * (mean_baseline - mean_efe) / mean_baseline,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                }
            )
        indices = range(start, len(comparison_rows))
        adjusted = holm([comparison_rows[index]["p_value"] for index in indices])
        for index, value in zip(indices, adjusted):
            comparison_rows[index]["holm_p"] = value
    return pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def online_table8(online: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ONLINE_METHODS:
        group = online[online["method"] == method]
        n = len(group)
        target = int(group["target_correct"].sum())
        attempted = int(group["attempted"].sum())
        grasp = int(group["grasp_success"].sum())
        full = int(group["full_task_success"].sum())
        prevented = int(group["wrong_prevented"].sum())
        target_low, target_high = wilson(target, n)
        full_low, full_high = wilson(full, n)
        grasp_low, grasp_high = wilson(grasp, attempted)
        asked_questions = group.loc[group["asked"], "question_count"]
        rows.append(
            {
                "method": method,
                "n": n,
                "target_correct": target,
                "target_accuracy_percent": 100.0 * target / n,
                "target_wilson_low_percent": 100.0 * target_low,
                "target_wilson_high_percent": 100.0 * target_high,
                "attempted": attempted,
                "grasp_success": grasp,
                "grasp_success_percent_of_attempts": 100.0 * grasp / attempted,
                "grasp_wilson_low_percent": 100.0 * grasp_low,
                "grasp_wilson_high_percent": 100.0 * grasp_high,
                "full_task_success": full,
                "full_task_success_percent": 100.0 * full / n,
                "full_task_wilson_low_percent": 100.0 * full_low,
                "full_task_wilson_high_percent": 100.0 * full_high,
                "wrong_prevented": prevented,
                "mean_questions_asked_trials": float(asked_questions.mean()) if len(asked_questions) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def sign_flip_p(
    differences: np.ndarray,
    rng: np.random.Generator,
    max_exact: int = 65_536,
    iterations: int = 20_000,
) -> tuple[float, str]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return np.nan, "all_zero_or_empty"
    observed = abs(float(values.mean()))
    total = 2 ** len(values)
    if total <= max_exact:
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
            statistic = abs(float((values * np.asarray(signs)).mean()))
            extreme += int(statistic >= observed - 1e-12)
        return float(extreme / total), "exact_sign_flip"
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, len(values)), replace=True)
    statistics = np.abs((signs * values).mean(axis=1))
    p_value = (np.count_nonzero(statistics >= observed - 1e-12) + 1) / (iterations + 1)
    return float(p_value), "monte_carlo_sign_flip"


def _consume_online_cluster_bootstrap_rng(online: pd.DataFrame, rng: np.random.Generator) -> None:
    # The manuscript analysis generated method-level cluster-bootstrap intervals
    # before its paired tests. Preserve that deterministic random stream so the
    # Monte Carlo p-values match the archived Table 9 outputs exactly.
    scene_count = online["scene_id"].nunique()
    for _ in range(ONLINE_BOOTSTRAP_ITERATIONS):
        rng.integers(0, scene_count, size=scene_count)


def _online_paired_row(
    online: pd.DataFrame,
    outcome: str,
    left: str,
    right: str,
    context: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    frame = online if context == "overall" else online[online["instruction_type"] == context]
    grouped = (
        frame[frame["method"].isin([left, right])]
        .groupby(["scene_id", "method"])[outcome]
        .mean()
        .reset_index()
    )
    paired = grouped.pivot(index="scene_id", columns="method", values=outcome).dropna()
    differences = (paired[left] - paired[right]).to_numpy(dtype=float)
    draws = rng.choice(
        differences,
        size=(ONLINE_BOOTSTRAP_ITERATIONS, len(differences)),
        replace=True,
    ).mean(axis=1)
    low, high = (float(value) for value in np.percentile(draws, [2.5, 97.5]))
    p_value, test = sign_flip_p(differences, rng)
    return {
        "context": context,
        "outcome": outcome,
        "method_A": left,
        "method_B": right,
        "n_scenes": int(len(paired)),
        "scene_mean_A": float(paired[left].mean()),
        "scene_mean_B": float(paired[right].mean()),
        "difference_percentage_points": 100.0 * float(differences.mean()),
        "ci_low_percentage_points": 100.0 * low,
        "ci_high_percentage_points": 100.0 * high,
        "p_value": p_value,
        "test": test,
    }


def online_table9(online: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(ONLINE_RANDOM_SEED)
    _consume_online_cluster_bootstrap_rng(online, rng)
    comparisons = [
        ("A_interactive_vs_noninteractive", "proposed_efe", "top_score"),
        ("A_interactive_vs_noninteractive", "proposed_efe", "random_candidate"),
        ("A_interactive_vs_noninteractive", "vlm_best_question", "top_score"),
        ("A_interactive_vs_noninteractive", "vlm_best_question", "random_candidate"),
        ("B_efe_vs_interactive", "proposed_efe", "vlm_best_question"),
    ]
    target_rows = []
    task_rows = []
    for context in ["overall", "clear", "ambiguous", "partial"]:
        for family, left, right in comparisons:
            target = _online_paired_row(online, "target_correct", left, right, context, rng)
            target["family"] = family
            target_rows.append(target)
            task = _online_paired_row(online, "full_task_success", left, right, context, rng)
            task["family"] = family
            task_rows.append(task)

    for rows in [target_rows, task_rows]:
        frame = pd.DataFrame(rows)
        for (_, family), indices in frame.groupby(["context", "family"]).groups.items():
            adjusted = holm(frame.loc[indices, "p_value"].tolist())
            for index, value in zip(indices, adjusted):
                rows[index]["holm_p"] = value

    selected = []
    for rows in [target_rows, task_rows]:
        for row in rows:
            if row["context"] != "overall" or row["method_A"] != "proposed_efe":
                continue
            selected.append(
                {
                    "outcome": row["outcome"],
                    "comparison": f"{row['method_A']} vs {row['method_B']}",
                    "n_scenes": row["n_scenes"],
                    "scene_mean_proposed_efe_percent": 100.0 * float(row["scene_mean_A"]),
                    "scene_mean_baseline_percent": 100.0 * float(row["scene_mean_B"]),
                    "difference_percentage_points": row["difference_percentage_points"],
                    "ci_low_percentage_points": row["ci_low_percentage_points"],
                    "ci_high_percentage_points": row["ci_high_percentage_points"],
                    "p_value": row["p_value"],
                    "holm_p": row["holm_p"],
                    "test": row["test"],
                }
            )
    return pd.DataFrame(selected)


def write(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUT / filename, index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    offline = pd.read_csv(DATA / "offline_trials.csv")
    online = pd.read_csv(DATA / "online_trials.csv")
    for frame in (offline, online):
        frame["target_correct"] = as_bool(frame["target_correct"])
        frame["asked"] = as_bool(frame["asked"])
    for column in ["attempted", "grasp_success", "full_task_success", "wrong_prevented"]:
        online[column] = as_bool(online[column])

    write(offline_table5(offline), "table5_offline_descriptive.csv")
    write(offline_table6(offline), "table6_offline_gee_odds_ratios.csv")
    question_summary, question_comparisons = question_table7(offline)
    write(question_summary, "table7_question_count_summary.csv")
    write(question_comparisons, "table7_question_count_comparisons.csv")
    write(online_table8(online), "table8_online_descriptive.csv")
    write(online_table9(online), "table9_online_comparisons.csv")
    print(f"Wrote manuscript-aligned Tables 5-9 to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
