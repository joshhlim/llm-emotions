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
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return ""
    return f"{float(value):.6f}"


def _json_ready(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats into JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    return value


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of flat records. Floats use six decimal places."""
    if not rows:
        raise ValueError(f"No rows to write for {path.name}.")
    header = list(rows[0])
    body: list[list[object]] = []
    for row in rows:
        rendered: list[object] = []
        for key in header:
            value = row[key]
            if isinstance(value, (bool, np.bool_)):
                rendered.append(int(bool(value)))
            elif isinstance(value, (float, np.floating)):
                rendered.append(_fmt(float(value)))
            elif isinstance(value, (int, np.integer)):
                rendered.append(int(value))
            else:
                rendered.append("" if value is None else value)
        body.append(rendered)
    _write_csv(path, header, body)


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


def save_phase02(
    config: ExperimentConfig,
    *,
    metrics: dict[str, Any],
    emotions: list[str],
    cosine_at_layer: dict[str, dict[str, np.ndarray]],
    transfer_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    effect_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    stacks: dict[str, Any],
    best_layer: int,
    figures: dict[str, Figure],
) -> Path:
    """Write the phase-02 bundle and return the output directory."""
    output_dir = config.output_dir()
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(_json_ready(config.to_dict()), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cosine_records: list[dict[str, Any]] = []
    for dataset, stages in cosine_at_layer.items():
        for stage, matrix in stages.items():
            for index, emotion in enumerate(emotions):
                record: dict[str, Any] = {
                    "dataset": dataset,
                    "stage": stage,
                    "emotion": emotion,
                }
                for column, other in enumerate(emotions):
                    record[other] = float(matrix[index, column])
                cosine_records.append(record)
    write_records(output_dir / "cosine_similarity.csv", cosine_records)
    write_records(output_dir / "transfer_accuracy.csv", transfer_rows)
    write_records(output_dir / "scenario_scores.csv", score_rows)
    write_records(output_dir / "scenario_effects.csv", effect_rows)
    write_records(output_dir / "scenario_deltas.csv", delta_rows)

    payload: dict[str, np.ndarray] = {
        "emotions": np.array(emotions),
        "best_layer": np.array(best_layer),
    }
    for name, stack in stacks.items():
        payload[f"{name}_raw"] = np.asarray(stack.raw)
        payload[f"{name}_mean_centered"] = np.asarray(stack.mean_centered)
        payload[f"{name}_cleaned"] = np.asarray(stack.cleaned)
        payload[f"{name}_affect"] = np.asarray(stack.affect)
    np.savez_compressed(output_dir / "directions.npz", **payload)

    for name, figure in figures.items():
        save_figure(figure, figure_dir / f"{name}.png")
    return output_dir
