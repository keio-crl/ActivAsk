from __future__ import annotations

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


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return float(center - margin), float(center + margin)


def bootstrap_mean_ci(values: np.ndarray, iterations: int = 10000, seed: int = 2026) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def pct(value: float) -> float:
    return float(value * 100.0)


def offline_table5(offline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in OFFLINE_METHODS:
        group = offline[offline["method"] == method]
        n = len(group)
        successes = int(group["target_correct"].sum())
        asked = int(group["asked"].sum())
        lo, hi = wilson(successes, n)
        asked_q = group.loc[group["asked"], "question_count"]
        rows.append(
            {
                "method": method,
                "n": n,
                "correct": successes,
                "target_accuracy_percent": pct(successes / n),
                "wilson_ci_low_percent": pct(lo),
                "wilson_ci_high_percent": pct(hi),
                "asked_percent": pct(asked / n),
                "mean_questions_all_trials": float(group["question_count"].mean()),
                "mean_questions_asked_trials": float(asked_q.mean()) if len(asked_q) else 0.0,
                "mean_latency_s": float(group["latency"].mean()),
                "median_latency_s": float(group["latency"].median()),
            }
        )
    return pd.DataFrame(rows)


def offline_table6(offline: pd.DataFrame) -> pd.DataFrame:
    work = offline.copy()
    work["target_correct_int"] = work["target_correct"].astype(int)
    model = smf.gee(
        "target_correct_int ~ C(method, Treatment(reference='proposed_efe'))"
        " + C(instruction_type) + C(scene_family) + candidate_count",
        groups="scene_id",
        data=work,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    )
    result = model.fit()
    params = result.params
    cov = result.cov_params()

    def contrast(left: str, right: str) -> tuple[float, float, float]:
        vec = pd.Series(0.0, index=params.index)
        for method, sign in ((left, 1.0), (right, -1.0)):
            key = f"C(method, Treatment(reference='proposed_efe'))[T.{method}]"
            if key in vec.index:
                vec[key] += sign
        log_or = float(vec @ params)
        se = float(np.sqrt(vec @ cov @ vec))
        p = float(2 * stats.norm.sf(abs(log_or / se))) if se > 0 else float("nan")
        return log_or, se, p

    comparisons = [
        ("proposed_efe", "top_score"),
        ("proposed_efe", "random_candidate"),
        ("proposed_efe", "vlm_direct"),
        ("vlm_best_question", "vlm_direct"),
    ]
    rows = []
    p_values = []
    for left, right in comparisons:
        log_or, se, p = contrast(left, right)
        p_values.append(p)
        rows.append(
            {
                "comparison": f"{left} vs {right}",
                "odds_ratio": float(np.exp(log_or)),
                "log_odds_ratio": log_or,
                "std_error": se,
                "p_value": p,
            }
        )
    for row, p_adj in zip(rows, holm(p_values)):
        row["holm_p"] = p_adj
    return pd.DataFrame(rows)


def question_table7(offline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for context, frame in [
        ("all_asked_trials", offline[offline["asked"]]),
        ("ambiguous_asked_trials", offline[(offline["asked"]) & (offline["instruction_type"] == "ambiguous")]),
    ]:
        for method in INTERACTIVE_METHODS:
            group = frame[frame["method"] == method]["question_count"].to_numpy(dtype=float)
            lo, hi = bootstrap_mean_ci(group, seed=4100 + len(rows))
            rows.append(
                {
                    "context": context,
                    "method": method,
                    "n": int(len(group)),
                    "mean_questions": float(np.mean(group)) if len(group) else float("nan"),
                    "median_questions": float(np.median(group)) if len(group) else float("nan"),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                }
            )

    comparisons = []
    p_values = []
    for context, frame in [
        ("all_asked_trials", offline[offline["asked"]]),
        ("ambiguous_asked_trials", offline[(offline["asked"]) & (offline["instruction_type"] == "ambiguous")]),
    ]:
        efe = frame[frame["method"] == "proposed_efe"]["question_count"].to_numpy(dtype=float)
        for baseline in ["first_question", "random_question", "vlm_best_question"]:
            base = frame[frame["method"] == baseline]["question_count"].to_numpy(dtype=float)
            p = float(stats.mannwhitneyu(efe, base, alternative="two-sided").pvalue)
            p_values.append(p)
            diff = float(np.mean(efe) - np.mean(base))
            comparisons.append(
                {
                    "context": context,
                    "comparison": f"proposed_efe vs {baseline}",
                    "mean_questions_proposed_efe": float(np.mean(efe)),
                    "mean_questions_baseline": float(np.mean(base)),
                    "mean_difference": diff,
                    "percent_reduction": float((-diff / np.mean(base)) * 100.0),
                    "p_value": p,
                }
            )
    for row, p_adj in zip(comparisons, holm(p_values)):
        row["holm_p"] = p_adj
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


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
        tlo, thi = wilson(target, n)
        flo, fhi = wilson(full, n)
        glo, ghi = wilson(grasp, attempted)
        asked_q = group.loc[group["asked"], "question_count"]
        rows.append(
            {
                "method": method,
                "n": n,
                "target_correct": target,
                "target_accuracy_percent": pct(target / n),
                "target_wilson_low_percent": pct(tlo),
                "target_wilson_high_percent": pct(thi),
                "attempted": attempted,
                "grasp_success": grasp,
                "grasp_success_percent_of_attempts": pct(grasp / attempted) if attempted else float("nan"),
                "grasp_wilson_low_percent": pct(glo),
                "grasp_wilson_high_percent": pct(ghi),
                "full_task_success": full,
                "full_task_success_percent": pct(full / n),
                "full_task_wilson_low_percent": pct(flo),
                "full_task_wilson_high_percent": pct(fhi),
                "wrong_prevented": prevented,
                "mean_questions_asked_trials": float(asked_q.mean()) if len(asked_q) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def online_table9(online: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p_values = []
    for outcome in ["target_correct", "full_task_success"]:
        for baseline in ["top_score", "random_candidate", "vlm_best_question"]:
            efe = online[online["method"] == "proposed_efe"][outcome].astype(float).to_numpy()
            base = online[online["method"] == baseline][outcome].astype(float).to_numpy()
            p = float(stats.mannwhitneyu(efe, base, alternative="two-sided").pvalue)
            p_values.append(p)
            rows.append(
                {
                    "outcome": outcome,
                    "comparison": f"proposed_efe vs {baseline}",
                    "proposed_efe_percent": pct(float(np.mean(efe))),
                    "baseline_percent": pct(float(np.mean(base))),
                    "difference_percentage_points": pct(float(np.mean(efe) - np.mean(base))),
                    "p_value": p,
                }
            )
    for row, p_adj in zip(rows, holm(p_values)):
        row["holm_p"] = p_adj
    return pd.DataFrame(rows)


def write(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUT / filename, index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    offline = pd.read_csv(DATA / "offline_trials.csv")
    online = pd.read_csv(DATA / "online_trials.csv")

    for frame in (offline, online):
        frame["target_correct"] = as_bool(frame["target_correct"])
        frame["asked"] = as_bool(frame["asked"])
    for col in ["attempted", "grasp_success", "full_task_success", "wrong_prevented"]:
        online[col] = as_bool(online[col])

    write(offline_table5(offline), "table5_offline_descriptive.csv")
    write(offline_table6(offline), "table6_offline_gee_odds_ratios.csv")
    q_summary, q_tests = question_table7(offline)
    write(q_summary, "table7_question_count_summary.csv")
    write(q_tests, "table7_question_count_comparisons.csv")
    write(online_table8(online), "table8_online_descriptive.csv")
    write(online_table9(online), "table9_online_comparisons.csv")
    print(f"Wrote tables to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
