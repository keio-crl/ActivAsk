from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/calibration/threshold_sweep.csv"
OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA).sort_values("tau")
    by_tau = df.groupby("tau", as_index=False).agg(
        recall=("recall", "max"),
        precision=("precision", "max"),
        f1=("f1", "max"),
        false_positives_per_image=("false_positives_per_image", "min"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(by_tau["tau"], by_tau["recall"], label="recall")
    axes[0].plot(by_tau["tau"], by_tau["precision"], label="precision")
    axes[0].plot(by_tau["tau"], by_tau["f1"], label="F1")
    axes[0].axvline(0.38, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("metric")
    axes[0].legend()

    axes[1].plot(by_tau["tau"], by_tau["false_positives_per_image"], color="tab:red")
    axes[1].axvline(0.38, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("false positives per image")

    out = OUT / "figure_a1_threshold_sweep.png"
    fig.savefig(out, dpi=200)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
