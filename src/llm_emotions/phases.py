"""Phase catalogue.

Phase 01 is implemented in :mod:`llm_emotions.experiment`. The later phases are
entry points with a question and no runner. Calling them raises
:class:`PhaseNotImplemented` and writes nothing.
"""

from __future__ import annotations

from llm_emotions.config import ExperimentConfig

PHASES: dict[str, dict[str, str]] = {
    "phase01_directions": {
        "title": "Emotion directions",
        "notebook": "notebooks/01_emotion_directions.ipynb",
        "question": (
            "Can contrastive first-person sentences yield linear directions and probes "
            "for emotion-like states in the residual stream?"
        ),
        "status": "implemented",
    },
    "phase02_scenarios": {
        "title": "Scenarios",
        "notebook": "notebooks/02_scenarios.ipynb",
        "question": (
            "Once each direction has the shared emotion-versus-neutral component "
            "removed, do scenario prompts still shift that direction relative to a "
            "matched control?"
        ),
        "status": "implemented",
    },
    "phase03_performance": {
        "title": "Performance",
        "notebook": "notebooks/03_performance.ipynb",
        "question": (
            "Does emotional framing change accuracy on a reasoning task, for example "
            "a GSM8K subset?"
        ),
        "status": "stub",
    },
    "phase04_steering": {
        "title": "Steering",
        "notebook": "notebooks/04_steering.ipynb",
        "question": (
            "Can steering an emotion direction up or down change task performance, "
            "compared with prompt-only emotional framing?"
        ),
        "status": "stub",
    },
}


class PhaseNotImplemented(Exception):
    """A phase entry point was called before that phase exists. No results were written."""


def _refuse(experiment: str) -> None:
    phase = PHASES.get(experiment)
    if phase is None:
        raise PhaseNotImplemented(f"{experiment} is not a known phase.\nNo results were written.")
    raise PhaseNotImplemented(
        f"{experiment} is not implemented.\nQuestion: {phase['question']}\nNo results were written."
    )


def run_phase02_scenarios(config: ExperimentConfig):
    """Scenario gaps on cleaned directions. A stub config still writes nothing."""
    if config.status != "implemented":
        _refuse("phase02_scenarios")
    from llm_emotions.phase02 import run_phase02

    return run_phase02(config)


def run_phase03_performance(config: ExperimentConfig) -> None:
    """Future phase: accuracy under emotional framing. Writes nothing."""
    del config
    _refuse("phase03_performance")


def run_phase04_steering(config: ExperimentConfig) -> None:
    """Future phase: task accuracy under steering. Writes nothing."""
    del config
    _refuse("phase04_steering")
