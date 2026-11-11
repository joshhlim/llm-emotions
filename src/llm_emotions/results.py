"""Write phase-01 metrics, tables, and figures under ``results/``."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from llm_emotions.caveats import PROBE_CAVEAT, STEERING_NOTE
from llm_emotions.config import ExperimentConfig
from llm_emotions.data import Scenario
from llm_emotions.directions import DirectionFit, cosine_matrix, direction_matrix
from llm_emotions.version import __version__


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.6f}"


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _accuracy_rows(fit: DirectionFit, attribute: str) -> list[list[object]]:
    table: dict[str, np.ndarray] = getattr(fit, attribute)
    n_layers = len(table[fit.emotions[0]])
    rows: list[list[object]] = []
    for layer in range(n_layers):
        rows.append([layer, *[_fmt(table[emotion][layer]) for emotion in fit.emotions]])
    return rows


def runtime_versions() -> dict[str, str]:
    import matplotlib
    import sklearn
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "llm_emotions": __version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _own_best_layer(accuracy: np.ndarray) -> int:
    later = accuracy[1:]
    if np.isfinite(later).any():
        return int(np.nanargmax(later) + 1)
    return int(np.nanargmax(accuracy))


def save_figure(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")


def save_phase01(
    config: ExperimentConfig,
    fit: DirectionFit,
    scores: dict[str, dict[str, float]],
    scenarios: list[Scenario],
    *,
    residual_norm: float,
    steering: list[dict[str, Any]],
    pca_explained_variance_ratio: list[float],
    figures: dict[str, Figure],
    contrastive_datasets: list[str],
    scenario_datasets: list[str],
    device: str,
    dtype: str,
    n_layers: int,
    hidden_size: int,
) -> Path:
    """Write the phase-01 bundle and return the output directory."""
    output_dir = config.output_dir()
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    emotions = fit.emotions
    cosine = cosine_matrix(direction_matrix(fit, fit.best_layer))
    per_emotion = {
        emotion: {
            "n_pairs": fit.n_pairs[emotion],
            "probe_accuracy_at_best_layer": fit.probe_accuracy[emotion][fit.best_layer],
            "direction_accuracy_at_best_layer": fit.direction_accuracy[emotion][fit.best_layer],
            "best_layer_for_emotion": _own_best_layer(fit.probe_accuracy[emotion]),
        }
        for emotion in emotions
    }
    metrics = {
        "kind": "phase01_directions",
        "caveat": PROBE_CAVEAT,
        "experiment": config.experiment,
        "model_id": config.model_id,
        "seed": config.seed,
        "device": device,
        "dtype": dtype,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "layers_requested": config.layers,
        "best_layer": fit.best_layer,
        "mean_probe_accuracy_at_best_layer": fit.mean_probe_at(fit.best_layer),
        "mean_direction_accuracy_at_best_layer": fit.mean_direction_at(fit.best_layer),
        "per_emotion": per_emotion,
        "pca_explained_variance_ratio": [float(value) for value in pca_explained_variance_ratio],
        "residual_norm_at_best_layer": residual_norm,
        "steering": {
            "emotion": config.steering_emotion,
            "prompt": config.steering_prompt,
            "note": STEERING_NOTE,
            "samples": steering,
        },
        "data": {
            "contrastive_datasets": contrastive_datasets,
            "scenario_datasets": scenario_datasets,
            "emotions_dir": config.emotions_dir,
            "scenarios_path": config.scenarios_path,
        },
        "versions": runtime_versions(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "steering_samples.json").write_text(
        json.dumps(
            {"note": STEERING_NOTE, "samples": steering},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    _write_csv(
        output_dir / "probe_accuracy.csv",
        ["layer", *emotions],
        _accuracy_rows(fit, "probe_accuracy"),
    )
    _write_csv(
        output_dir / "direction_accuracy.csv",
        ["layer", *emotions],
        _accuracy_rows(fit, "direction_accuracy"),
    )
    _write_csv(
        output_dir / "scenario_scores.csv",
        ["scenario_id", "pair", "role", *emotions],
        [
            [
                scenario.id,
                scenario.pair,
                scenario.role,
                *[_fmt(scores[scenario.id][emotion]) for emotion in emotions],
            ]
            for scenario in scenarios
        ],
    )
    _write_csv(
        output_dir / "cosine_similarity.csv",
        ["emotion", *emotions],
        [
            [emotion, *[_fmt(cosine[row, column]) for column in range(len(emotions))]]
            for row, emotion in enumerate(emotions)
        ],
    )

    np.savez_compressed(
        output_dir / "directions.npz",
        emotions=np.array(emotions),
        directions=np.stack([fit.directions[emotion] for emotion in emotions]),
        probe_accuracy=np.stack([fit.probe_accuracy[emotion] for emotion in emotions]),
        direction_accuracy=np.stack([fit.direction_accuracy[emotion] for emotion in emotions]),
        best_layer=np.array(fit.best_layer),
        residual_norm=np.array(residual_norm),
    )

    for name, figure in figures.items():
        save_figure(figure, figure_dir / f"{name}.png")
    return output_dir
