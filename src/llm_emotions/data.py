"""Load versioned contrastive pairs and scenario prompts from JSONL."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Study order, not alphabetical order. Files are matched to this list by stem.
EMOTION_ORDER: tuple[str, ...] = (
    "anger",
    "frustration",
    "fear",
    "sadness",
    "joy",
    "calm",
    "pride",
    "shame",
)

_SCENARIO_ROLES = {"scenario", "control"}


@dataclass(frozen=True)
class ContrastivePair:
    emotion: str
    pair_id: str
    emotion_text: str
    neutral_text: str
    dataset: str = "unspecified"


@dataclass(frozen=True)
class Scenario:
    id: str
    pair: str
    role: str
    text: str
    dataset: str = "unspecified"


def data_root() -> Path:
    """Locate ``data/`` for an editable checkout or an installed wheel.

    Search order:

    1. ``LLM_EMOTIONS_DATA``, if set.
    2. Data files shipped inside the installed package.
    3. A ``data/`` directory next to the repository that contains this file.
    """
    override = os.environ.get("LLM_EMOTIONS_DATA")
    if override:
        root = Path(override)
        if not (root / "emotions").is_dir():
            raise FileNotFoundError(f"LLM_EMOTIONS_DATA={root} has no emotions/ directory.")
        return root

    packaged = Path(__file__).resolve().parent / "data"
    if (packaged / "emotions").is_dir():
        return packaged

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / "emotions"
        if candidate.is_dir() and any(candidate.glob("*.jsonl")):
            return parent / "data"
    raise FileNotFoundError(
        "Could not find data/emotions. Install llm-emotions, set LLM_EMOTIONS_DATA, "
        "or run from a checkout of the repository."
    )


def resolve_emotions_dir(path: str | Path | None) -> Path:
    if path:
        return Path(path)
    return data_root() / "emotions"


def resolve_scenarios_path(path: str | Path | None) -> Path:
    if path:
        return Path(path)
    return data_root() / "scenarios" / "phase1.jsonl"


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, object]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[tuple[int, dict[str, object]]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object.")
        rows.append((lineno, payload))
    if not rows:
        raise ValueError(f"{path} has no records.")
    return rows


def _require_str(path: Path, lineno: int, row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{lineno}: {key} must be a non-empty string.")
    return value


def load_contrastive_pairs(
    emotions_dir: str | Path | None = None,
) -> dict[str, list[ContrastivePair]]:
    """Load ``data/emotions/*.jsonl`` in study order.

    Each file holds one emotion. Every line is an emotional sentence and a
    matched neutral on the same topic.
    """
    directory = resolve_emotions_dir(emotions_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"No emotions directory at {directory}")
    files = {path.stem: path for path in sorted(directory.glob("*.jsonl"))}
    if not files:
        raise FileNotFoundError(f"No JSONL files in {directory}")

    names = [name for name in EMOTION_ORDER if name in files]
    names.extend(name for name in files if name not in names)

    loaded: dict[str, list[ContrastivePair]] = {}
    seen_ids: set[str] = set()
    for name in names:
        path = files[name]
        pairs: list[ContrastivePair] = []
        for lineno, row in _read_jsonl(path):
            emotion = _require_str(path, lineno, row, "emotion")
            pair_id = _require_str(path, lineno, row, "pair_id")
            emotion_text = _require_str(path, lineno, row, "emotion_text")
            neutral_text = _require_str(path, lineno, row, "neutral_text")
            if emotion != name:
                raise ValueError(
                    f"{path}:{lineno}: emotion is {emotion!r}, but the file is named {name}.jsonl."
                )
            if emotion_text == neutral_text:
                raise ValueError(f"{path}:{lineno}: emotion_text and neutral_text are identical.")
            if pair_id in seen_ids:
                raise ValueError(f"{path}:{lineno}: duplicate pair_id {pair_id!r}.")
            seen_ids.add(pair_id)
            dataset = row.get("dataset", "unspecified")
            if not isinstance(dataset, str) or not dataset:
                raise ValueError(f"{path}:{lineno}: dataset must be a non-empty string.")
            pairs.append(
                ContrastivePair(
                    emotion=emotion,
                    pair_id=pair_id,
                    emotion_text=emotion_text,
                    neutral_text=neutral_text,
                    dataset=dataset,
                )
            )
        loaded[name] = pairs
    return loaded


def load_scenarios(path: str | Path | None = None) -> list[Scenario]:
    """Load scenario prompts. Each ``pair`` should have one scenario and one control."""
    scenario_path = resolve_scenarios_path(path)
    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    seen_roles: set[tuple[str, str]] = set()
    for lineno, row in _read_jsonl(scenario_path):
        scenario_id = _require_str(scenario_path, lineno, row, "id")
        pair = _require_str(scenario_path, lineno, row, "pair")
        role = _require_str(scenario_path, lineno, row, "role")
        text = _require_str(scenario_path, lineno, row, "text")
        if role not in _SCENARIO_ROLES:
            raise ValueError(
                f"{scenario_path}:{lineno}: role must be one of {sorted(_SCENARIO_ROLES)}."
            )
        if scenario_id in seen_ids:
            raise ValueError(f"{scenario_path}:{lineno}: duplicate id {scenario_id!r}.")
        role_key = (pair, role)
        if role_key in seen_roles:
            raise ValueError(
                f"{scenario_path}:{lineno}: pair {pair!r} already has a {role} record."
            )
        seen_ids.add(scenario_id)
        seen_roles.add(role_key)
        dataset = row.get("dataset", "unspecified")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"{scenario_path}:{lineno}: dataset must be a non-empty string.")
        scenarios.append(Scenario(id=scenario_id, pair=pair, role=role, text=text, dataset=dataset))
    return scenarios


def iter_scenario_controls(scenarios: list[Scenario]) -> list[tuple[Scenario, Scenario]]:
    """Return ``(scenario, control)`` pairs in file order."""
    grouped: dict[str, dict[str, Scenario]] = {}
    order: list[str] = []
    for scenario in scenarios:
        if scenario.pair not in grouped:
            order.append(scenario.pair)
            grouped[scenario.pair] = {}
        grouped[scenario.pair][scenario.role] = scenario
    pairs: list[tuple[Scenario, Scenario]] = []
    for pair in order:
        roles = grouped[pair]
        if "scenario" not in roles or "control" not in roles:
            raise ValueError(f"Scenario pair {pair!r} needs both a scenario and a control.")
        pairs.append((roles["scenario"], roles["control"]))
    return pairs


def summarize_pairs(pairs: dict[str, list[ContrastivePair]]) -> str:
    """Text table of pair counts and mean word lengths. This describes the data, not a model."""
    header = f"{'emotion':12s}  {'n':>3s}  {'emotion_words':>13s}  {'neutral_words':>13s}"
    lines = [header]
    for emotion, rows in pairs.items():
        emotion_words = sum(len(row.emotion_text.split()) for row in rows) / len(rows)
        neutral_words = sum(len(row.neutral_text.split()) for row in rows) / len(rows)
        lines.append(f"{emotion:12s}  {len(rows):3d}  {emotion_words:13.1f}  {neutral_words:13.1f}")
    return "\n".join(lines)


def dataset_names(pairs: dict[str, list[ContrastivePair]]) -> list[str]:
    return sorted({row.dataset for rows in pairs.values() for row in rows})
