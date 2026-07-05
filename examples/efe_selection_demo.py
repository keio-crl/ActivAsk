from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activask import Candidate, CandidateState, QuestionPartition, answer_question, choose_question, tokenize_terms


def main() -> None:
    candidates = (
        Candidate("cand_001", "cup", 0.72),
        Candidate("cand_002", "cup", 0.61),
        Candidate("cand_003", "cup", 0.58),
        Candidate("cand_004", "cup", 0.55),
    )
    state = CandidateState.from_candidates(candidates)
    questions = [
        QuestionPartition(1, "Is it orange?", ("cand_001", "cand_002"), ("cand_003", "cand_004")),
        QuestionPartition(2, "Is it on the left side?", ("cand_001",), ("cand_002", "cand_003", "cand_004")),
        QuestionPartition(3, "Does it have a handle?", ("cand_003",), ("cand_001", "cand_002", "cand_004")),
        QuestionPartition(4, "Is it near the plate?", ("cand_002", "cand_004"), ("cand_001", "cand_003")),
    ]
    selected = choose_question(questions, state, policy="proposed_efe")
    next_state = answer_question(state, selected, "yes")
    terms = tokenize_terms("Pick up the left cup.")
    print("Candidates:", ", ".join(state.plausible))
    print("Instruction noun:", terms.noun)
    print("Instruction modifiers:", ", ".join(terms.modifiers))
    print("Selected question:", selected.text)
    print("Yes candidates:", ", ".join(selected.yes_candidates))
    print("No candidates:", ", ".join(selected.no_candidates))
    print("Plausible after yes:", ", ".join(next_state.plausible))


if __name__ == "__main__":
    main()
