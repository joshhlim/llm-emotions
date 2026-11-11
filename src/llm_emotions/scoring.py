"""Project scenario prompts onto emotion directions."""

from __future__ import annotations

from typing import Any

import numpy as np

from llm_emotions.activations import collect_hidden_states
from llm_emotions.data import Scenario


def score_scenarios(
    model: Any,
    tokenizer: Any,
    scenarios: list[Scenario],
    directions: dict[str, np.ndarray],
    layer: int,
    *,
    batch_size: int = 8,
) -> dict[str, dict[str, float]]:
    """Dot the last-token residual at ``layer`` with each unit direction.

    Scores are in arbitrary units. Compare a scenario with its matched control.
    The absolute value is not a probability and not an emotion intensity.
    """
    if not scenarios:
        raise ValueError("scenarios is empty.")
    hidden = collect_hidden_states(
        model,
        tokenizer,
        [scenario.text for scenario in scenarios],
        batch_size=batch_size,
    )
    emotions = list(directions)
    scores: dict[str, dict[str, float]] = {}
    for index, scenario in enumerate(scenarios):
        residual = hidden[index, layer]
        scores[scenario.id] = {
            emotion: float(residual @ directions[emotion][layer]) for emotion in emotions
        }
    return scores
