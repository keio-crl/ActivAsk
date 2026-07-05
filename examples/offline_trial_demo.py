from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activask import Candidate, choose_candidate_baseline


def main() -> None:
    df = pd.read_csv(ROOT / "data/offline_trials.csv")
    example = df.iloc[0]
    print("Offline trial example")
    print(f"scene_id: {example.scene_id}")
    print(f"instruction_type: {example.instruction_type}")
    print(f"instruction_text: {example.instruction_text}")
    print(f"method: {example.method}")
    print(f"target_correct: {example.target_correct}")

    candidates = [
        Candidate("cand_001", "cup", 0.42),
        Candidate("cand_002", "cup", 0.71),
        Candidate("cand_003", "cup", 0.63),
    ]
    selected = choose_candidate_baseline(candidates, method="top_score")
    print(f"top_score_demo_selection: {selected.candidate_id}")


if __name__ == "__main__":
    main()
