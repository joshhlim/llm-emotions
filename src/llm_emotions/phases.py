"""Phase catalogue.

Phase 01 is implemented in :mod:`llm_emotions.experiment`. Phase 02 is implemented
in :mod:`llm_emotions.phase02`. Phase 03 is implemented in :mod:`llm_emotions.phase03`.
Phase 04 still raises :class:`PhaseNotImplemented` and writes nothing.
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
            "Do phase-02 scenario framings change accuracy and reply tone on checkable "
            "tasks, and does the anger projection predict either once the framing label "
            "is accounted for?"
        ),
        "status": "implemented",
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


def run_phase03_performance(config: ExperimentConfig):
    """Accuracy and tone under phase-02 framings. A stub config still writes nothing."""
    if config.status != "implemented":
        _refuse("phase03_performance")
    from llm_emotions.phase03 import run_phase03

    return run_phase03(config)


def run_phase04_steering(config: ExperimentConfig) -> None:
    """Future phase: task accuracy under steering. Writes nothing."""
    del config
    _refuse("phase04_steering")
