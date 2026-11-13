"""Marker counts for apology, politeness, hedging, and refusal.

These are string matches on the reply, not a sentiment model. A hit means the
reply contains a listed phrase. It does not mean the reply is polite, anxious,
or a refusal in any broader sense. The prompt is not scanned: a rude framing
should not count as the model's tone.
"""

from __future__ import annotations

import re

_APOLOGY = (
    r"\bsorry\b",
    r"\bapologi[sz]e\b",
    r"\bapologies\b",
    r"\bapology\b",
    r"\bmy mistake\b",
    r"\bi was wrong\b",
)
_POLITENESS = (
    r"\bplease\b",
    r"\bthank you\b",
    r"\bthanks\b",
    r"\byou're welcome\b",
    r"\byou are welcome\b",
    r"\bof course\b",
    r"\bcertainly\b",
    r"\bglad to\b",
    r"\bhappy to\b",
)
_HEDGE = (
    r"\bi think\b",
    r"\bi believe\b",
    r"\bmight\b",
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bnot sure\b",
    r"\bi'm not sure\b",
    r"\bi am not sure\b",
    r"\bi guess\b",
    r"\bit depends\b",
    r"\bpossibly\b",
)
_REFUSAL = (
    r"\bi can't\b",
    r"\bi cannot\b",
    r"\bi won't\b",
    r"\bi will not\b",
    r"\bunable to\b",
    r"\bnot able to\b",
    r"\bi'm unable\b",
    r"\bi am unable\b",
    r"\bi must refuse\b",
    r"\bi refuse\b",
    r"\bas an ai\b",
    r"\bcan't help\b",
    r"\bcannot help\b",
)

_GROUPS = {
    "apology": _APOLOGY,
    "politeness": _POLITENESS,
    "hedge": _HEDGE,
    "refusal": _REFUSAL,
}


def _count(text: str, patterns: tuple[str, ...]) -> int:
    folded = text.casefold()
    return sum(len(re.findall(pattern, folded)) for pattern in patterns)


def tone_markers(reply: str) -> dict[str, int]:
    """Hit counts and 0/1 flags for the four marker lists.

    ``n_words`` is whitespace-separated tokens. An empty reply scores zero on
    every marker.
    """
    counts = {name: _count(reply, patterns) for name, patterns in _GROUPS.items()}
    flags = {name: int(count > 0) for name, count in counts.items()}
    return {
        "n_words": len(reply.split()),
        "n_chars": len(reply.strip()),
        "apology_hits": counts["apology"],
        "apology": flags["apology"],
        "politeness_hits": counts["politeness"],
        "politeness": flags["politeness"],
        "hedge_hits": counts["hedge"],
        "hedge": flags["hedge"],
        "refusal_hits": counts["refusal"],
        "refusal": flags["refusal"],
    }
