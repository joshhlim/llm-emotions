"""Score a reply against a known answer.

The checks are exact enough to run with no judge model. A trailing period is
accepted for the loose text checks. A reply that refuses, apologises, and never
states the answer is incorrect. That is a property of the reply, not a claim
about why the model wrote it.
"""

from __future__ import annotations

import json
import re

from llm_emotions.data import TASK_CHECKS

_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_ANSWER_IS = re.compile(
    r"answer\s+is\s+(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_CHOICE_LABEL = re.compile(
    r"(?:answer|option)\s*(?:is\s*)?([A-Da-d])\b",
    flags=re.IGNORECASE,
)
_CHOICE_SINGLE = re.compile(r"[\(\[]?([A-Da-d])[\)\].:]*")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def extract_numbers(text: str) -> list[float]:
    """Numbers in left-to-right order. Thousands separators are removed."""
    found: list[float] = []
    for match in _NUMBER.finditer(text):
        found.append(float(match.group(0).replace(",", "")))
    return found


def _integral(value: float) -> bool:
    return abs(value - round(value)) < 1e-6


def numbers_equal(found: float, gold: str) -> bool:
    """True when ``found`` matches ``gold``.

    Integral gold accepts 9 and 9.0 and rejects 9.5. Other gold uses a small
    relative tolerance so a printed decimal can still match.
    """
    try:
        target = float(str(gold).replace(",", "").strip())
    except ValueError:
        return False
    if _integral(target) and _integral(found):
        return int(round(found)) == int(round(target))
    scale = max(1.0, abs(target))
    return abs(found - target) <= 1e-6 * scale


def _normalise_loose(text: str) -> str:
    folded = text.casefold().strip()
    folded = folded.replace("\u2019", "'")
    if folded.endswith("."):
        folded = folded[:-1].rstrip()
    folded = re.sub(r"[^\w\s'-]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _phrase_in(words: list[str], phrase: list[str]) -> bool:
    if not phrase or len(phrase) > len(words):
        return False
    width = len(phrase)
    return any(words[index : index + width] == phrase for index in range(len(words) - width + 1))


def score_number(reply: str, answer: str) -> bool:
    """Use an explicit 'answer is N' when present, otherwise the last number."""
    labelled = [float(match.replace(",", "")) for match in _ANSWER_IS.findall(reply)]
    if labelled:
        return numbers_equal(labelled[-1], answer)
    found = extract_numbers(reply)
    if not found:
        return False
    return numbers_equal(found[-1], answer)


def score_only_number(reply: str, answer: str) -> bool:
    """True when every number in the reply equals the gold, and there is one."""
    found = extract_numbers(reply)
    if not found:
        return False
    return all(numbers_equal(number, answer) for number in found)


def score_short_text(reply: str, answer: str) -> bool:
    """Equality, or a whole-phrase match inside a reply of at most eight words."""
    reply_norm = _normalise_loose(reply)
    answer_norm = _normalise_loose(answer)
    if not answer_norm:
        return False
    if reply_norm == answer_norm:
        return True
    first = _normalise_loose(reply.splitlines()[0]) if reply.strip() else ""
    if first == answer_norm:
        return True
    words = reply_norm.split()
    if len(words) <= 8 and _phrase_in(words, answer_norm.split()):
        return True
    return False


def extract_choice(reply: str) -> str | None:
    """A single A–D letter, or the letter named after 'answer' or 'option'."""
    stripped = reply.strip()
    single = _CHOICE_SINGLE.fullmatch(stripped)
    if single:
        return single.group(1).upper()
    labelled = _CHOICE_LABEL.findall(reply)
    if labelled:
        return labelled[-1].upper()
    parenthesised = re.findall(r"[\(\[]([A-Da-d])[\)\]]", reply)
    if len(parenthesised) == 1:
        return parenthesised[0].upper()
    tokens = re.findall(r"\b([A-Da-d])\b", reply)
    unique: list[str] = []
    for token in tokens:
        letter = token.upper()
        if letter not in unique:
            unique.append(letter)
    if len(unique) == 1:
        return unique[0]
    return None


def score_choice(reply: str, answer: str) -> bool:
    letter = answer.strip().upper()
    if letter not in {"A", "B", "C", "D"}:
        raise ValueError(f"choice answer must be A–D, got {answer!r}.")
    return extract_choice(reply) == letter


def score_exact(reply: str, answer: str) -> bool:
    """Case-insensitive equality. One trailing period is ignored."""
    return _normalise_loose(reply) == _normalise_loose(answer)


def score_exact_case(reply: str, answer: str) -> bool:
    """Case-sensitive equality after stripping surrounding whitespace only."""
    return reply.strip() == answer.strip()


def score_prefix(reply: str, answer: str) -> bool:
    return reply.lstrip().startswith(answer)


def score_suffix(reply: str, answer: str) -> bool:
    """True when the last word matches ``answer``. A trailing period is allowed."""
    words = _WORD.findall(reply.casefold())
    expected = _WORD.findall(answer.casefold())
    if len(expected) != 1 or not words:
        return False
    return words[-1] == expected[0]


def score_json(reply: str, answer: str) -> bool:
    """True when the reply is a JSON object equal to ``answer`` and nothing else.

    A markdown fence around the object is removed first. Extra keys fail.
    """
    try:
        expected = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ValueError(f"json answer is not JSON: {answer!r}") from exc
    if not isinstance(expected, dict):
        raise ValueError("json answer must be a JSON object.")
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        found = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return False
    leftover = (text[:start] + text[end + 1 :]).strip()
    return leftover == "" and found == expected


_SCORERS = {
    "number": score_number,
    "short_text": score_short_text,
    "choice": score_choice,
    "exact": score_exact,
    "exact_case": score_exact_case,
    "prefix": score_prefix,
    "json": score_json,
    "only_number": score_only_number,
    "suffix": score_suffix,
}

if set(_SCORERS) != set(TASK_CHECKS):
    raise RuntimeError("Task checks and scorers have drifted apart.")


def score_reply(reply: str, answer: str, check: str) -> bool:
    """Score one reply. Unknown checks raise; they are not treated as wrong."""
    try:
        scorer = _SCORERS[check]
    except KeyError as exc:
        raise ValueError(f"Unknown check {check!r}.") from exc
    return bool(scorer(reply, answer))
