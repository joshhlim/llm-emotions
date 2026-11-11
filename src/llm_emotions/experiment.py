"""Run phase 01 from a config and write ``results/<experiment>/<model>/``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm_emotions.activations import collect_hidden_states
from llm_emotions.caveats import PROBE_CAVEAT
from llm_emotions.config import ExperimentConfig
from llm_emotions.data import (
    Scenario,
    dataset_names,
    load_contrastive_pairs,
    load_scenarios,
    resolve_emotions_dir,
    resolve_scenarios_path,
)
from llm_emotions.directions import DirectionFit, fit_directions_and_probes, set_seed
from llm_emotions.model import load_model_and_tokenizer, module_device
from llm_emotions.plotting import plot_direction_pca, plot_probe_accuracy, plot_scenario_scores
from llm_emotions.reporting import format_probe_table, format_score_table
from llm_emotions.results import save_phase01
from llm_emotions.scoring import score_scenarios
from llm_emotions.steering import generate_with_steering, mean_residual_norm

logger = logging.getLogger("llm_emotions")


def configure_logging() -> None:
    """Log this package's progress as plain lines, without changing the root logger."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@dataclass
class Phase01Result:
    config: ExperimentConfig
    fit: DirectionFit
    scores: dict[str, dict[str, float]]
    scenarios: list[Scenario]
    residual_norm: float
    steering: list[dict[str, Any]]
    pca_explained_variance_ratio: list[float]
    output_dir: str | None
    device: str
    dtype: str
    n_layers: int
    hidden_size: int
    figures: dict[str, Any] = field(default_factory=dict)


def run_phase01(
    config: ExperimentConfig,
    *,
    model: Any = None,
    tokenizer: Any = None,
    save: bool = True,
    do_steering: bool = True,
) -> Phase01Result:
    """Extract activations, fit directions, score scenarios, and optionally steer.

    Pass ``model`` and ``tokenizer`` together to skip ``from_pretrained``. Tests
    use that path with a tiny random network. A normal run leaves both unset and
    loads ``config.model_id``.
    """
    if (model is None) ^ (tokenizer is None):
        raise ValueError("Pass both model and tokenizer, or neither.")
    set_seed(config.seed)
    logger.info(
        "experiment=%s model=%s seed=%s",
        config.experiment,
        config.model_id,
        config.seed,
    )

    if model is None:
        loaded = load_model_and_tokenizer(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.device
        dtype = loaded.dtype
        n_layers = loaded.n_layers
        hidden_size = loaded.hidden_size
    else:
        device = str(module_device(model))
        dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        n_layers = int(model.config.num_hidden_layers)
        hidden_size = int(model.config.hidden_size)

    logger.info(
        "device=%s dtype=%s layers=%s hidden=%s",
        device,
        dtype,
        n_layers,
        hidden_size,
    )

    emotions_dir = resolve_emotions_dir(config.emotions_dir)
    scenarios_path = resolve_scenarios_path(config.scenarios_path)
    config.emotions_dir = str(emotions_dir)
    config.scenarios_path = str(scenarios_path)

    pairs = load_contrastive_pairs(emotions_dir)
    scenarios = load_scenarios(scenarios_path)
    logger.info("Loaded %s emotions, %s scenario prompts.", len(pairs), len(scenarios))

    activations: dict[str, dict[str, np.ndarray]] = {}
    for emotion, rows in pairs.items():
        logger.info("Extracting %s (%s pairs)", emotion, len(rows))
        activations[emotion] = {
            "emotion": collect_hidden_states(
                model,
                tokenizer,
                [row.emotion_text for row in rows],
                batch_size=config.batch_size,
            ),
            "neutral": collect_hidden_states(
                model,
                tokenizer,
                [row.neutral_text for row in rows],
                batch_size=config.batch_size,
            ),
        }

    fit = fit_directions_and_probes(
        activations,
        seed=config.seed,
        n_splits=config.n_splits,
        probe_C=config.probe_C,
        probe_max_iter=config.probe_max_iter,
        layers=config.layers,
    )
    logger.info("%s", format_probe_table(fit))
    logger.info("%s", PROBE_CAVEAT)

    if config.steering_emotion not in fit.emotions:
        raise ValueError(f"steering_emotion {config.steering_emotion!r} is not in {fit.emotions}.")

    logger.info("Scoring %s scenario prompts at layer %s.", len(scenarios), fit.best_layer)
    scores = score_scenarios(
        model,
        tokenizer,
        scenarios,
        fit.directions,
        fit.best_layer,
        batch_size=config.batch_size,
    )
    logger.info(
        "%s",
        format_score_table(scores, fit.emotions, [scenario.id for scenario in scenarios]),
    )

    residual_norm = mean_residual_norm(activations, fit.best_layer)
    steering: list[dict[str, Any]] = []
    if do_steering:
        scaled = fit.directions[config.steering_emotion][fit.best_layer] * residual_norm
        logger.info(
            "Steering %s at layer %s. Mean residual L2 norm: %.1f",
            config.steering_emotion,
            fit.best_layer,
            residual_norm,
        )
        for strength in config.steering_strengths:
            text = generate_with_steering(
                model,
                tokenizer,
                config.steering_prompt,
                layer=fit.best_layer,
                direction=scaled,
                strength=float(strength),
                max_new_tokens=config.max_new_tokens,
            )
            steering.append({"strength": float(strength), "text": text})
            logger.info("--- strength %+.3f ---\n%s", strength, text)

    probe_fig = plot_probe_accuracy(fit)
    pca = plot_direction_pca(fit, seed=config.seed)
    scenario_fig = plot_scenario_scores(scores, scenarios, fit.emotions, fit.best_layer)
    figures = {
        "probe_accuracy": probe_fig,
        "direction_pca": pca.figure,
        "scenario_scores": scenario_fig,
    }
    variance = [float(value) for value in pca.explained_variance_ratio]

    output_dir: str | None = None
    if save:
        written = save_phase01(
            config,
            fit,
            scores,
            scenarios,
            residual_norm=residual_norm,
            steering=steering,
            pca_explained_variance_ratio=variance,
            figures=figures,
            contrastive_datasets=dataset_names(pairs),
            scenario_datasets=sorted({scenario.dataset for scenario in scenarios}),
            device=device,
            dtype=dtype,
            n_layers=n_layers,
            hidden_size=hidden_size,
        )
        output_dir = str(written)
        logger.info("Wrote %s", written)

    return Phase01Result(
        config=config,
        fit=fit,
        scores=scores,
        scenarios=scenarios,
        residual_norm=residual_norm,
        steering=steering,
        pca_explained_variance_ratio=variance,
        output_dir=output_dir,
        device=device,
        dtype=dtype,
        n_layers=n_layers,
        hidden_size=hidden_size,
        figures=figures,
    )
