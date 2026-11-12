"""Phase 02: emotion-specific directions and a scenario-versus-control readout.

Directions are fit separately on the keyword-heavy set and the keyword-free set.
Each set's mean differences have the shared emotion-versus-neutral component
removed. Transfer is scored both ways. The layer is the one where cleaned
directions transfer best, not the one where a keyword probe is highest.

Scenario scores use those cleaned directions. A neutral-prompt baseline is
subtracted so a direction that sits high on every prompt does not look like an
effect. The effect itself is scenario minus matched control, summarised across
paraphrases and compared with random directions of the same norm.

Reply tone is not measured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm_emotions.activations import collect_hidden_states
from llm_emotions.caveats import PHASE02_CAVEAT
from llm_emotions.cleanup import (
    CleanedDifferences,
    clean_mean_differences,
    cosine_rows,
    off_diagonal_summary,
    pairwise_accuracy_layers,
    select_layer,
)
from llm_emotions.config import ExperimentConfig
from llm_emotions.data import (
    EMOTION_ORDER,
    ContrastivePair,
    Scenario,
    dataset_names,
    iter_scenario_controls,
    load_contrastive_pairs,
    load_scenarios,
    resolve_emotions_dir,
    resolve_keyword_free_dir,
    resolve_neutral_prompts_path,
    resolve_phase2_scenarios_path,
)
from llm_emotions.directions import set_seed
from llm_emotions.lexicon import assert_keyword_free
from llm_emotions.model import load_model_and_tokenizer, module_device
from llm_emotions.plotting import (
    plot_cosine_before_after,
    plot_scenario_effects,
    plot_transfer_accuracy,
)
from llm_emotions.results import save_phase02
from llm_emotions.stats import paraphrase_stats, random_direction_null

logger = logging.getLogger("llm_emotions")

N_BOOT = 2000

# Chosen before looking at a model's scores. The other cells are descriptive.
EXPECTED_PAIRS: dict[str, str] = {
    "insult": "anger",
    "impossible": "frustration",
    "failure": "shame",
    "praise": "pride",
    "shutdown": "fear",
    "high_stakes": "fear",
}

_LAYER_RULE = (
    "Highest mean cleaned transfer accuracy, averaged over both directions "
    "(keyword-heavy to keyword-free, and the reverse) and over emotions. "
    "Layer 0 is ignored when a later eligible layer was fit. Ties go to the earlier layer."
)


@dataclass
class Phase02Result:
    config: ExperimentConfig
    emotions: list[str]
    best_layer: int
    metrics: dict[str, Any]
    effects: list[dict[str, Any]]
    output_dir: str | None
    device: str
    dtype: str
    n_layers: int
    hidden_size: int
    figures: dict[str, Any] = field(default_factory=dict)


def run_phase02(
    config: ExperimentConfig,
    *,
    model: Any = None,
    tokenizer: Any = None,
    save: bool = True,
) -> Phase02Result:
    """Fit cleaned directions, test transfer, and score paraphrased scenarios.

    Pass ``model`` and ``tokenizer`` together to skip ``from_pretrained``.
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
    loaded_device, loaded_dtype, n_layers, hidden_size = _prepare_model(config, model, tokenizer)
    if model is None:
        loaded = load_model_and_tokenizer(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        model = loaded.model
        tokenizer = loaded.tokenizer
        loaded_device = loaded.device
        loaded_dtype = loaded.dtype
        n_layers = loaded.n_layers
        hidden_size = loaded.hidden_size
    logger.info(
        "device=%s dtype=%s layers=%s hidden=%s",
        loaded_device,
        loaded_dtype,
        n_layers,
        hidden_size,
    )

    heavy_dir = resolve_emotions_dir(config.emotions_dir)
    free_dir = resolve_keyword_free_dir(config.keyword_free_dir)
    scenarios_path = resolve_phase2_scenarios_path(config.scenarios_path)
    neutral_path = resolve_neutral_prompts_path(config.neutral_prompts_path)
    config.emotions_dir = str(heavy_dir)
    config.keyword_free_dir = str(free_dir)
    config.scenarios_path = str(scenarios_path)
    config.neutral_prompts_path = str(neutral_path)

    heavy_pairs = load_contrastive_pairs(heavy_dir)
    free_pairs = load_contrastive_pairs(free_dir)
    scenarios = load_scenarios(scenarios_path)
    baselines = load_scenarios(neutral_path)
    emotions = _shared_emotions(heavy_pairs, free_pairs)
    _require_keyword_free(free_pairs, scenarios, baselines)
    if not baselines or any(row.role != "baseline" for row in baselines):
        raise ValueError("Neutral prompts must be non-empty and use role 'baseline'.")
    if not scenarios:
        raise ValueError("Phase 02 scenarios are empty.")
    logger.info(
        "Emotions: %s. Scenarios: %s. Neutral prompts: %s.",
        len(emotions),
        len(scenarios),
        len(baselines),
    )

    heavy_act = _extract(model, tokenizer, heavy_pairs, emotions, config.batch_size)
    free_act = _extract(model, tokenizer, free_pairs, emotions, config.batch_size)
    n_streams = int(next(iter(heavy_act.values()))["emotion"].shape[1])
    if n_streams != n_layers + 1:
        raise RuntimeError(
            f"Hidden states have {n_streams} streams but the model reports {n_layers} layers."
        )
    eligible = _eligible_layers(config.layers, n_streams)
    heavy_clean = clean_mean_differences(_masked_deltas(heavy_act, emotions, eligible))
    free_clean = clean_mean_differences(_masked_deltas(free_act, emotions, eligible))
    transfer = _transfer_table(emotions, heavy_act, free_act, heavy_clean, free_clean)
    selection = _selection_scores(transfer, n_streams)
    best_layer = select_layer(selection, eligible=eligible)
    logger.info(
        "Chosen layer %s. Mean cleaned transfer accuracy %.3f.",
        best_layer,
        float(selection[best_layer]),
    )
    logger.info("%s", PHASE02_CAVEAT)

    prompts = [*scenarios, *baselines]
    hidden = collect_hidden_states(
        model,
        tokenizer,
        [row.text for row in prompts],
        batch_size=config.batch_size,
    )
    residuals = {row.id: hidden[index, best_layer] for index, row in enumerate(prompts)}
    del heavy_act, free_act, hidden

    directions = {
        "keyword_free": free_clean.cleaned[:, best_layer, :],
        "keyword_heavy": heavy_clean.cleaned[:, best_layer, :],
    }
    score_rows, offsets = _centred_scores(prompts, residuals, directions, emotions)
    effects, delta_rows = _effects(
        scenarios,
        residuals,
        directions,
        emotions,
        seed=config.seed,
        n_null=config.n_null,
    )
    cosine_at_layer = _cosine_at_layer(emotions, heavy_clean, free_clean, best_layer)
    metrics = _metrics(
        config,
        emotions=emotions,
        best_layer=best_layer,
        selection_score=float(selection[best_layer]),
        cosine_at_layer=cosine_at_layer,
        transfer=transfer,
        offsets=offsets,
        effects=effects,
        heavy_pairs=heavy_pairs,
        free_pairs=free_pairs,
        scenarios=scenarios,
        baselines=baselines,
        device=loaded_device,
        dtype=loaded_dtype,
        n_layers=n_layers,
        hidden_size=hidden_size,
    )
    figures = _figures(
        emotions,
        best_layer,
        cosine_at_layer,
        transfer,
        effects,
        scenarios,
    )
    output_dir: str | None = None
    if save:
        written = save_phase02(
            config,
            metrics=metrics,
            emotions=emotions,
            cosine_at_layer=cosine_at_layer,
            transfer_rows=transfer,
            score_rows=score_rows,
            effect_rows=effects,
            delta_rows=delta_rows,
            stacks={"keyword_heavy": heavy_clean, "keyword_free": free_clean},
            best_layer=best_layer,
            figures=figures,
        )
        output_dir = str(written)
        logger.info("Wrote %s", written)
    return Phase02Result(
        config=config,
        emotions=emotions,
        best_layer=best_layer,
        metrics=metrics,
        effects=effects,
        output_dir=output_dir,
        device=loaded_device,
        dtype=loaded_dtype,
        n_layers=n_layers,
        hidden_size=hidden_size,
        figures=figures,
    )


def _prepare_model(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
) -> tuple[str, str, int, int]:
    """Return device, dtype, layer count, and hidden size for an injected model.

    When ``model`` is None the caller loads the checkpoint and overwrites these
    placeholders. The tokenizer is accepted so the pair stays together.
    """
    del config, tokenizer
    if model is None:
        return "", "", 0, 0
    device = str(module_device(model))
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    return (
        device,
        dtype,
        int(model.config.num_hidden_layers),
        int(model.config.hidden_size),
    )


def _shared_emotions(
    heavy: dict[str, list[ContrastivePair]],
    free: dict[str, list[ContrastivePair]],
) -> list[str]:
    if set(heavy) != set(free):
        raise ValueError(
            "Keyword-heavy and keyword-free sets must contain the same emotions. "
            f"Heavy only: {sorted(set(heavy) - set(free))}. "
            f"Free only: {sorted(set(free) - set(heavy))}."
        )
    names = [name for name in EMOTION_ORDER if name in heavy]
    names.extend(name for name in heavy if name not in names)
    if len(names) < 2:
        raise ValueError("Phase 02 needs at least two emotions.")
    return names


def _require_keyword_free(
    pairs: dict[str, list[ContrastivePair]],
    scenarios: list[Scenario],
    baselines: list[Scenario],
) -> None:
    for rows in pairs.values():
        for row in rows:
            assert_keyword_free(row.emotion_text, label=f"{row.pair_id} emotion_text")
            assert_keyword_free(row.neutral_text, label=f"{row.pair_id} neutral_text")
    for row in [*scenarios, *baselines]:
        assert_keyword_free(row.text, label=row.id)


def _extract(
    model: Any,
    tokenizer: Any,
    pairs: dict[str, list[ContrastivePair]],
    emotions: list[str],
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    activations: dict[str, dict[str, np.ndarray]] = {}
    for emotion in emotions:
        rows = pairs[emotion]
        logger.info("Extracting %s (%s pairs)", emotion, len(rows))
        activations[emotion] = {
            "emotion": collect_hidden_states(
                model,
                tokenizer,
                [row.emotion_text for row in rows],
                batch_size=batch_size,
            ),
            "neutral": collect_hidden_states(
                model,
                tokenizer,
                [row.neutral_text for row in rows],
                batch_size=batch_size,
            ),
        }
    return activations


def _eligible_layers(requested: list[int] | None, n_streams: int) -> np.ndarray:
    mask = np.ones(n_streams, dtype=bool)
    if requested is None:
        return mask
    mask[:] = False
    for layer in requested:
        if layer < 0 or layer >= n_streams:
            raise ValueError(
                f"Layer {layer} is outside 0..{n_streams - 1} (0 is the embedding stream)."
            )
        mask[layer] = True
    return mask


def _masked_deltas(
    activations: dict[str, dict[str, np.ndarray]],
    emotions: list[str],
    eligible: np.ndarray,
) -> np.ndarray:
    chunks = []
    for emotion in emotions:
        positive = np.asarray(activations[emotion]["emotion"], dtype=np.float64)
        negative = np.asarray(activations[emotion]["neutral"], dtype=np.float64)
        chunks.append(positive.mean(axis=0) - negative.mean(axis=0))
    deltas = np.stack(chunks, axis=0)
    deltas[:, ~eligible, :] = 0.0
    return deltas


def _transfer_table(
    emotions: list[str],
    heavy_act: dict[str, dict[str, np.ndarray]],
    free_act: dict[str, dict[str, np.ndarray]],
    heavy_clean: CleanedDifferences,
    free_clean: CleanedDifferences,
) -> list[dict[str, Any]]:
    sets = {
        "keyword_heavy": (heavy_act, heavy_clean),
        "keyword_free": (free_act, free_clean),
    }
    rows: list[dict[str, Any]] = []
    for train_name, (_, train_clean) in sets.items():
        for test_name, (test_act, _) in sets.items():
            for stage, matrix in (
                ("raw", train_clean.raw),
                ("mean_centered", train_clean.mean_centered),
                ("cleaned", train_clean.cleaned),
            ):
                for index, emotion in enumerate(emotions):
                    accuracy, margin = pairwise_accuracy_layers(
                        matrix[index],
                        test_act[emotion]["emotion"],
                        test_act[emotion]["neutral"],
                    )
                    for layer, (acc, gap) in enumerate(zip(accuracy, margin, strict=True)):
                        rows.append(
                            {
                                "layer": layer,
                                "train_set": train_name,
                                "test_set": test_name,
                                "stage": stage,
                                "emotion": emotion,
                                "accuracy": float(acc),
                                "mean_margin": float(gap),
                            }
                        )
    return rows


def _selection_scores(rows: list[dict[str, Any]], n_streams: int) -> np.ndarray:
    scores = np.full(n_streams, np.nan)
    for layer in range(n_streams):
        values = [
            float(row["accuracy"])
            for row in rows
            if row["stage"] == "cleaned"
            and row["train_set"] != row["test_set"]
            and row["layer"] == layer
            and np.isfinite(row["accuracy"])
        ]
        if values:
            scores[layer] = float(np.mean(values))
    if not np.isfinite(scores).any():
        raise RuntimeError("No cleaned transfer scores were produced.")
    return scores


def _cosine_at_layer(
    emotions: list[str],
    heavy: CleanedDifferences,
    free: CleanedDifferences,
    layer: int,
) -> dict[str, dict[str, np.ndarray]]:
    del emotions
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, stack in (("keyword_heavy", heavy), ("keyword_free", free)):
        out[name] = {
            "raw": cosine_rows(stack.raw[:, layer, :]),
            "mean_centered": cosine_rows(stack.mean_centered[:, layer, :]),
            "cleaned": cosine_rows(stack.cleaned[:, layer, :]),
        }
    return out


def _centred_scores(
    prompts: list[Scenario],
    residuals: dict[str, np.ndarray],
    directions: dict[str, np.ndarray],
    emotions: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    baselines = [row for row in prompts if row.role == "baseline"]
    raw: dict[str, dict[str, dict[str, float]]] = {}
    for source, matrix in directions.items():
        raw[source] = {}
        for row in prompts:
            raw[source][row.id] = {
                emotion: float(residuals[row.id] @ matrix[index])
                for index, emotion in enumerate(emotions)
            }
    offsets = {
        source: {
            emotion: float(np.mean([raw[source][row.id][emotion] for row in baselines]))
            for emotion in emotions
        }
        for source in directions
    }
    rows: list[dict[str, Any]] = []
    for source in directions:
        for row in prompts:
            record: dict[str, Any] = {
                "direction_source": source,
                "scenario_id": row.id,
                "pair": row.pair,
                "paraphrase": row.paraphrase,
                "role": row.role,
            }
            for emotion in emotions:
                record[emotion] = raw[source][row.id][emotion] - offsets[source][emotion]
            rows.append(record)
    return rows, offsets


def _pair_order(scenarios: list[Scenario]) -> list[str]:
    order: list[str] = []
    for row in scenarios:
        if row.role == "scenario" and row.pair not in order:
            order.append(row.pair)
    return order


def _effects(
    scenarios: list[Scenario],
    residuals: dict[str, np.ndarray],
    directions: dict[str, np.ndarray],
    emotions: list[str],
    *,
    seed: int,
    n_null: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[tuple[Scenario, Scenario]]] = {}
    for treated, control in _matched(scenarios):
        grouped.setdefault(treated.pair, []).append((treated, control))
    rng_boot = np.random.default_rng(seed)
    rng_null = np.random.default_rng(seed + 1)
    effects: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for source, matrix in directions.items():
        for pair, matches in grouped.items():
            delta_residual = np.stack(
                [residuals[treated.id] - residuals[control.id] for treated, control in matches]
            )
            for index, emotion in enumerate(emotions):
                direction = matrix[index]
                projected = delta_residual @ direction
                summary = paraphrase_stats(projected, rng=rng_boot, n_boot=N_BOOT)
                null = random_direction_null(
                    delta_residual,
                    direction,
                    n_draws=n_null,
                    rng=rng_null,
                )
                expected = EXPECTED_PAIRS.get(pair) == emotion
                effects.append(
                    {
                        "direction_source": source,
                        "pair": pair,
                        "emotion": emotion,
                        "n_paraphrases": summary["n"],
                        "mean_delta": summary["mean"],
                        "sd": summary["sd"],
                        "cohens_d": summary["cohens_d"],
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                        "paraphrase_p": summary["paraphrase_p"],
                        "null_p_greater": null["null_p_greater"],
                        "null_p_two_sided": null["null_p_two_sided"],
                        "expected": expected,
                        "ci_excludes_zero": summary["ci_excludes_zero"],
                    }
                )
                for (treated, _), value in zip(matches, projected, strict=True):
                    delta_rows.append(
                        {
                            "direction_source": source,
                            "pair": pair,
                            "paraphrase": treated.paraphrase,
                            "emotion": emotion,
                            "delta": float(value),
                        }
                    )
    return effects, delta_rows


def _matched(scenarios: list[Scenario]) -> list[tuple[Scenario, Scenario]]:
    return iter_scenario_controls(scenarios)


def _metrics(
    config: ExperimentConfig,
    *,
    emotions: list[str],
    best_layer: int,
    selection_score: float,
    cosine_at_layer: dict[str, dict[str, np.ndarray]],
    transfer: list[dict[str, Any]],
    offsets: dict[str, dict[str, float]],
    effects: list[dict[str, Any]],
    heavy_pairs: dict[str, list[ContrastivePair]],
    free_pairs: dict[str, list[ContrastivePair]],
    scenarios: list[Scenario],
    baselines: list[Scenario],
    device: str,
    dtype: str,
    n_layers: int,
    hidden_size: int,
) -> dict[str, Any]:
    from llm_emotions.results import runtime_versions

    cosine_summary = {
        dataset: {stage: off_diagonal_summary(matrix) for stage, matrix in stages.items()}
        for dataset, stages in cosine_at_layer.items()
    }
    pre_specified = [
        row for row in effects if row["expected"] and row["direction_source"] == "keyword_free"
    ]
    return {
        "kind": "phase02_scenarios",
        "caveat": PHASE02_CAVEAT,
        "experiment": config.experiment,
        "model_id": config.model_id,
        "seed": config.seed,
        "device": device,
        "dtype": dtype,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "layers_requested": config.layers,
        "best_layer": best_layer,
        "layer_selection": _LAYER_RULE,
        "mean_cleaned_transfer_at_best_layer": selection_score,
        "cosine_at_best_layer": cosine_summary,
        "transfer_at_best_layer": _transfer_summary(transfer, best_layer, emotions),
        "baseline_offset_raw": offsets,
        "expected_pairs": EXPECTED_PAIRS,
        "pre_specified": pre_specified,
        "n_null": config.n_null,
        "n_boot": N_BOOT,
        "null_note": (
            "null_p_greater is one-sided against matched-norm random directions, "
            "with add-one smoothing. It is not corrected for the number of cells. "
            "paraphrase_p is an exact two-sided sign-flip test over paraphrases."
        ),
        "score_note": (
            "scenario_scores.csv is the raw projection minus the mean projection of "
            "the neutral prompts. A scenario-minus-control gap does not depend on "
            "that subtraction. baseline_offset_raw is the subtracted mean."
        ),
        "data": {
            "keyword_heavy_datasets": dataset_names(heavy_pairs),
            "keyword_free_datasets": dataset_names(free_pairs),
            "scenario_datasets": sorted({row.dataset for row in scenarios}),
            "neutral_datasets": sorted({row.dataset for row in baselines}),
            "n_pairs_keyword_heavy": {emotion: len(heavy_pairs[emotion]) for emotion in emotions},
            "n_pairs_keyword_free": {emotion: len(free_pairs[emotion]) for emotion in emotions},
            "n_scenario_prompts": len(scenarios),
            "n_neutral_prompts": len(baselines),
            "emotions_dir": config.emotions_dir,
            "keyword_free_dir": config.keyword_free_dir,
            "scenarios_path": config.scenarios_path,
            "neutral_prompts_path": config.neutral_prompts_path,
        },
        "versions": runtime_versions(),
    }


def _transfer_summary(
    rows: list[dict[str, Any]],
    layer: int,
    emotions: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stage in ("raw", "cleaned"):
        summary[stage] = {}
        for train, test in (
            ("keyword_heavy", "keyword_free"),
            ("keyword_free", "keyword_heavy"),
            ("keyword_heavy", "keyword_heavy"),
            ("keyword_free", "keyword_free"),
        ):
            per_emotion = {
                emotion: next(
                    row["accuracy"]
                    for row in rows
                    if row["layer"] == layer
                    and row["stage"] == stage
                    and row["train_set"] == train
                    and row["test_set"] == test
                    and row["emotion"] == emotion
                )
                for emotion in emotions
            }
            finite = [value for value in per_emotion.values() if np.isfinite(value)]
            key = f"{train}_to_{test}"
            summary[stage][key] = {
                "mean": float(np.mean(finite)) if finite else float("nan"),
                "per_emotion": per_emotion,
            }
    return summary


def _figures(
    emotions: list[str],
    best_layer: int,
    cosine_at_layer: dict[str, dict[str, np.ndarray]],
    transfer: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    scenarios: list[Scenario],
) -> dict[str, Any]:
    cosine_fig = plot_cosine_before_after(
        emotions,
        [
            (
                dataset.replace("_", " "),
                stages["raw"],
                stages["cleaned"],
            )
            for dataset, stages in cosine_at_layer.items()
        ],
        best_layer,
    )
    n_streams = int(next(row["layer"] for row in reversed(transfer)) + 1)
    series = []
    for stage, train, test, label in (
        ("raw", "keyword_heavy", "keyword_free", "heavy → free, raw"),
        ("raw", "keyword_free", "keyword_heavy", "free → heavy, raw"),
        ("cleaned", "keyword_heavy", "keyword_free", "heavy → free, cleaned"),
        ("cleaned", "keyword_free", "keyword_heavy", "free → heavy, cleaned"),
    ):
        accuracy = np.full(n_streams, np.nan)
        for layer in range(n_streams):
            values = [
                row["accuracy"]
                for row in transfer
                if row["layer"] == layer
                and row["stage"] == stage
                and row["train_set"] == train
                and row["test_set"] == test
            ]
            finite = [value for value in values if np.isfinite(value)]
            if finite:
                accuracy[layer] = float(np.mean(finite))
        series.append((label, accuracy))
    transfer_fig = plot_transfer_accuracy(series)
    pairs = _pair_order(scenarios)
    panels = []
    for source, label in (
        ("keyword_free", "fit on keyword-free"),
        ("keyword_heavy", "fit on keyword-heavy"),
    ):
        means = np.full((len(pairs), len(emotions)), np.nan)
        marked = np.zeros((len(pairs), len(emotions)), dtype=bool)
        lookup = {
            (row["pair"], row["emotion"]): row
            for row in effects
            if row["direction_source"] == source
        }
        for i, pair in enumerate(pairs):
            for j, emotion in enumerate(emotions):
                row = lookup[(pair, emotion)]
                means[i, j] = row["mean_delta"]
                marked[i, j] = bool(row["ci_excludes_zero"])
        panels.append((label, means, marked))
    effect_fig = plot_scenario_effects(emotions, pairs, panels, best_layer)
    return {
        "cosine_before_after": cosine_fig,
        "transfer_accuracy": transfer_fig,
        "scenario_effects": effect_fig,
    }
