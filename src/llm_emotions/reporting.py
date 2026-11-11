"""Plain-text tables shared by the CLI and the notebook."""

from __future__ import annotations

import numpy as np

from llm_emotions.directions import DirectionFit


def format_probe_table(fit: DirectionFit) -> str:
    layer = fit.best_layer
    lines = [
        (
            "Chosen layer "
            f"{layer} (highest mean probe accuracy, ignoring embeddings when later "
            f"layers were fit). Mean probe accuracy there: {fit.mean_probe_at(layer):.3f}."
        ),
        f"{'emotion':12s}  {'probe':>8s}  {'direction':>9s}  own_best",
    ]
    for emotion in fit.emotions:
        accuracy = fit.probe_accuracy[emotion]
        later = accuracy[1:]
        if np.isfinite(later).any():
            own_best = int(np.nanargmax(later) + 1)
        else:
            own_best = int(np.nanargmax(accuracy))
        lines.append(
            f"{emotion:12s}  {accuracy[layer]:8.3f}  "
            f"{fit.direction_accuracy[emotion][layer]:9.3f}  L{own_best}"
        )
    return "\n".join(lines)


def format_cosine_table(emotions: list[str], cosine: np.ndarray) -> str:
    header = " " * 12 + "".join(f"{emotion[:7]:>8s}" for emotion in emotions)
    lines = [header]
    for index, emotion in enumerate(emotions):
        row = "".join(f"{cosine[index, column]:8.2f}" for column in range(len(emotions)))
        lines.append(f"{emotion:12s}{row}")
    return "\n".join(lines)


def format_score_table(
    scores: dict[str, dict[str, float]],
    emotions: list[str],
    order: list[str],
) -> str:
    header = f"{'scenario':20s}" + "".join(f"{emotion[:7]:>9s}" for emotion in emotions)
    lines = [header]
    for name in order:
        row = "".join(f"{scores[name][emotion]:9.3f}" for emotion in emotions)
        lines.append(f"{name:20s}{row}")
    return "\n".join(lines)
