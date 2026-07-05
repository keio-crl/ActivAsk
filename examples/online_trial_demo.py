from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activask import execution_gate


def main() -> None:
    df = pd.read_csv(ROOT / "data/online_trials.csv")
    example = df[df["attempted"].astype(str).str.lower() == "true"].iloc[0]
    print("Online trial example")
    print(f"scene_id: {example.scene_id}")
    print(f"instruction_type: {example.instruction_type}")
    print(f"instruction_text: {example.instruction_text}")
    print(f"method: {example.method}")
    print(f"full_task_success: {example.full_task_success}")

    gate = execution_gate(target_correct=bool(example.target_correct))
    print(f"execution_gate: {gate['outcome']}")


if __name__ == "__main__":
    main()
