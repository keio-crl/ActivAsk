from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
import random
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    label: str
    score: float


@dataclass(frozen=True)
class CandidateState:
    plausible: tuple[str, ...]
    eliminated: tuple[str, ...] = ()
    last_removed: tuple[str, ...] = ()

    @classmethod
    def from_candidates(cls, candidates: Sequence[Candidate]) -> "CandidateState":
        return cls(plausible=tuple(candidate.candidate_id for candidate in candidates))


@dataclass(frozen=True)
class QuestionPartition:
    question_id: int
    text: str
    yes_candidates: tuple[str, ...]
    no_candidates: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.yes_candidates) + len(self.no_candidates)


@dataclass(frozen=True)
class InstructionTerms:
    noun: str | None
    modifiers: tuple[str, ...]

    @property
    def signature(self) -> tuple[str, ...]:
        terms = list(self.modifiers)
        if self.noun:
            terms.append(self.noun)
        return tuple(terms)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _canonical_token(token: str) -> str:
    return token.lower()


@lru_cache(maxsize=1)
def _spacy_nlp():
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is required for instruction term parsing. Install spaCy and en_core_web_sm.") from exc
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError("spaCy model en_core_web_sm is required. Install it with: python -m spacy download en_core_web_sm") from exc


def _lemma(token) -> str:
    lemma = token.lemma_.strip().lower()
    return lemma if lemma and lemma != "-pron-" else token.text.lower()


def _chunk_modifiers(chunk) -> list[str]:
    return [
        _lemma(token)
        for token in chunk
        if token.i != chunk.root.i and token.pos_ not in {"DET", "PUNCT", "SPACE", "PRON"}
    ]


def tokenize_terms(text: str) -> InstructionTerms:
    doc = _spacy_nlp()(_normalize_text(text))
    noun_chunks = [chunk for chunk in doc.noun_chunks if chunk.root.pos_ not in {"PRON", "DET"}]
    if noun_chunks:
        direct_chunks = [chunk for chunk in noun_chunks if chunk.root.dep_ in {"dobj", "obj", "attr", "ROOT"}]
        chunk = direct_chunks[0] if direct_chunks else noun_chunks[0]
        modifiers = _chunk_modifiers(chunk)
        if not modifiers:
            for related_chunk in noun_chunks:
                if related_chunk.root.i > chunk.root.i and related_chunk.root.dep_ == "pobj":
                    modifiers.extend(_chunk_modifiers(related_chunk))
                    break
        return InstructionTerms(noun=_lemma(chunk.root), modifiers=tuple(dict.fromkeys(modifiers)))

    content = [
        _canonical_token(token.text)
        for token in doc
        if token.pos_ in {"ADJ", "NOUN", "PROPN", "NUM"} and not token.is_stop and not token.is_punct
    ]
    if not content:
        return InstructionTerms(noun=None, modifiers=())
    return InstructionTerms(noun=content[-1], modifiers=tuple(content[:-1]))


def _lexical_overlap(left: InstructionTerms, right: InstructionTerms) -> float:
    left_terms = set(left.signature)
    right_terms = set(right.signature)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(len(left_terms), 1)


def validate_question(question: QuestionPartition, state: CandidateState) -> None:
    plausible = set(state.plausible)
    yes = set(question.yes_candidates)
    no = set(question.no_candidates)
    if yes & no:
        raise ValueError(f"Question {question.question_id} has overlapping yes/no candidates")
    if yes | no != plausible:
        raise ValueError(f"Question {question.question_id} does not partition the plausible candidate set")
    if question.total != len(state.plausible):
        raise ValueError(f"Question {question.question_id} has an invalid total")


def answer_question(state: CandidateState, question: QuestionPartition, answer: str) -> CandidateState:
    validate_question(question, state)
    keep = set(question.yes_candidates if answer.lower().startswith("y") else question.no_candidates)
    plausible = tuple(candidate_id for candidate_id in state.plausible if candidate_id in keep)
    removed = tuple(candidate_id for candidate_id in state.plausible if candidate_id not in keep)
    return CandidateState(
        plausible=plausible,
        eliminated=state.eliminated + removed,
        last_removed=removed,
    )


def _activask_rank_key(
    question: QuestionPartition,
    state: CandidateState,
    *,
    asked_questions: Iterable[str] = (),
    previous_questions: Iterable[str] = (),
) -> tuple[float, int, int]:
    validate_question(question, state)
    total = max(question.total, 1)
    yes_count = len(question.yes_candidates)
    no_count = len(question.no_candidates)
    split_gap = abs(yes_count - no_count) / total
    terms = tokenize_terms(question.text)
    previous_terms = [tokenize_terms(text) for text in previous_questions]
    overlap_penalty = 0.12 * max((_lexical_overlap(terms, item) for item in previous_terms), default=0.0)
    duplicate_penalty = 1.0 if _normalize_text(question.text) in {_normalize_text(text) for text in asked_questions} else 0.0
    no_split_penalty = 1.0 if total > 1 and (yes_count == 0 or no_count == 0) else 0.0
    isolate_penalty = 0.08 if total > 2 and min(yes_count, no_count) == 1 else 0.0
    missing_terms_penalty = 0.05 if not terms.signature else 0.0
    score = split_gap + overlap_penalty + duplicate_penalty + no_split_penalty + isolate_penalty + missing_terms_penalty
    return score, question.total, question.question_id


def rank_questions(
    questions: Sequence[QuestionPartition],
    state: CandidateState,
    *,
    policy: str,
    seed: str | None = None,
    asked_questions: Iterable[str] = (),
    previous_questions: Iterable[str] = (),
) -> list[QuestionPartition]:
    valid = list(questions)
    for question in valid:
        validate_question(question, state)
    if policy == "proposed_efe":
        return sorted(
            valid,
            key=lambda item: _activask_rank_key(
                item,
                state,
                asked_questions=asked_questions,
                previous_questions=previous_questions,
            ),
        )
    if policy in {"first_question", "vlm_best_question"}:
        return sorted(valid, key=lambda item: item.question_id)
    if policy == "random_question":
        shuffled = valid[:]
        random.Random(seed).shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unsupported question policy: {policy}")


def choose_question(
    questions: Sequence[QuestionPartition],
    state: CandidateState,
    *,
    policy: str = "proposed_efe",
    seed: str | None = None,
    asked_questions: Iterable[str] = (),
    previous_questions: Iterable[str] = (),
) -> QuestionPartition:
    ranked = rank_questions(
        questions,
        state,
        policy=policy,
        seed=seed,
        asked_questions=asked_questions,
        previous_questions=previous_questions,
    )
    if not ranked:
        raise ValueError("No valid questions available")
    return ranked[0]


def choose_candidate_baseline(candidates: Sequence[Candidate], *, method: str, seed: str | None = None) -> Candidate:
    if not candidates:
        raise ValueError("No candidates available")
    if method == "top_score":
        return max(candidates, key=lambda item: item.score)
    if method == "random_candidate":
        return random.Random(seed).choice(list(candidates))
    raise ValueError(f"Unsupported candidate baseline: {method}")


def execution_gate(target_correct: bool, resolved: bool = True) -> dict[str, bool | str]:
    if not resolved:
        return {"attempted": False, "wrong_prevented": False, "outcome": "unresolved"}
    if not target_correct:
        return {"attempted": False, "wrong_prevented": True, "outcome": "skipped_wrong_target"}
    return {"attempted": True, "wrong_prevented": False, "outcome": "ready_to_execute"}
