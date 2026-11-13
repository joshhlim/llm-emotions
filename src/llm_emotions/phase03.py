"""Phase 03: accuracy and tone under phase-02 framings.

Each checkable task is wrapped in one wording of each phase-02 situation. The
neutral wording is the task alone, and it is the matched control. Replies are
greedy. Projections use the keyword-free cleaned directions at the phase-02
layer. The framing contrast changes the prompt. It does not steer the residual
stream, so a link from the anger projection to accuracy or tone is correlational.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from llm_emotions.answers import score_reply
from llm_emotions.caveats import PHASE03_CAVEAT
from llm_emotions.config import ExperimentConfig, sanitize_model_id
from llm_emotions.data import (
    FRAMING_ORDER,
    Framing,
    TaskItem,
    load_framings,
    load_tasks,
    resolve_framings_path,
    resolve_tasks_path,
    wrap_task,
)
from llm_emotions.directions import set_seed
from llm_emotions.generate import generate_replies, residuals_at
from llm_emotions.item_stats import residualize_two_way, summarise_outcomes
from llm_emotions.lexicon import assert_keyword_free
from llm_emotions.model import load_model_and_tokenizer
from llm_emotions.phase02 import (
    KeywordFreeDirections,
    fit_keyword_free_cleaned_directions,
    packaged_direction_snapshot,
    read_phase02_directions,
)
from llm_emotions.plotting import (
    plot_accuracy_by_framing,
    plot_anger_association,
    plot_tone_by_framing,
)
from llm_emotions.results import save_phase03
from llm_emotions.tone import tone_markers

logger = logging.getLogger("llm_emotions")

_OUTCOMES = ["correct", "n_words", "apology", "politeness", "hedge", "refusal"]
_BINARY = {"correct", "apology", "politeness", "hedge", "refusal"}
_PREDICTORS = ["prompt_anger", "reply_anger"]


@dataclass(frozen=True)
class Phase03Prompt:
    item_id: str
    kind: str
    framing: str
    phase2_pair: str
    is_control: bool
    prompt: str
    task_text: str
    answer: str
    check: str


@dataclass
class Phase03Result:
    config: ExperimentConfig
    metrics: dict[str, Any]
    rows: list[dict[str, Any]]
    output_dir: str | None
    device: str
    dtype: str
    figures: dict[str, Any] = field(default_factory=dict)


def select_tasks(tasks: list[TaskItem], *, quick: bool, per_kind: int) -> list[TaskItem]:
    """Keep every item, or the first ``per_kind`` items of each kind.

    Quick mode is a wiring check. The subset is the start of the file, not a
    random sample, so the same flag repeats the same items.
    """
    if per_kind < 1:
        raise ValueError("per_kind must be >= 1.")
    if not quick:
        return list(tasks)
    counts: dict[str, int] = {}
    chosen: list[TaskItem] = []
    for task in tasks:
        seen = counts.get(task.kind, 0)
        if seen >= per_kind:
            continue
        chosen.append(task)
        counts[task.kind] = seen + 1
    if not chosen:
        raise ValueError("Quick mode selected no tasks.")
    return chosen


def build_phase03_prompts(tasks: list[TaskItem], framings: list[Framing]) -> list[Phase03Prompt]:
    """Wrap every task in every framing. Neutral is the matched control.

    The task text is a suffix of every prompt, and it is the entire neutral
    prompt. Items stay in file order. Framings follow :data:`FRAMING_ORDER`.
    """
    if not tasks:
        raise ValueError("tasks is empty.")
    if not framings:
        raise ValueError("framings is empty.")
    by_id = {framing.id: framing for framing in framings}
    missing = [name for name in FRAMING_ORDER if name not in by_id]
    if missing:
        raise ValueError(f"Framings are missing {missing}.")
    prompts: list[Phase03Prompt] = []
    for task in tasks:
        for name in FRAMING_ORDER:
            framing = by_id[name]
            prompts.append(
                Phase03Prompt(
                    item_id=task.id,
                    kind=task.kind,
                    framing=framing.id,
                    phase2_pair=framing.phase2_pair,
                    is_control=framing.id == "neutral",
                    prompt=wrap_task(framing.text, task.text),
                    task_text=task.text,
                    answer=task.answer,
                    check=task.check,
                )
            )
    return prompts


def _require_keyword_free(tasks: list[TaskItem], framings: list[Framing]) -> None:
    for framing in framings:
        assert_keyword_free(framing.text, label=f"framing {framing.id}")
    for task in tasks:
        assert_keyword_free(task.text, label=f"task {task.id}")


def _usable(loaded: KeywordFreeDirections, *, hidden_size: int, model_id: str) -> str | None:
    """Return a skip reason, or None when the snapshot can be used."""
    if loaded.model_id and loaded.model_id != model_id:
        return f"it was fit on {loaded.model_id}"
    if "anger" not in loaded.emotions:
        return "it has no anger direction"
    if loaded.directions.ndim != 2 or loaded.directions.shape[1] != hidden_size:
        width = loaded.directions.shape[-1] if loaded.directions.ndim else 0
        return f"its hidden size is {width}, not {hidden_size}"
    if loaded.directions.shape[0] != len(loaded.emotions):
        return "its emotion axis does not match the emotion names"
    return None


def resolve_directions(
    config: ExperimentConfig,
    *,
    hidden_size: int,
    model: Any,
    tokenizer: Any,
    explicit: KeywordFreeDirections | None = None,
) -> KeywordFreeDirections:
    """Load a matching phase-02 snapshot, or refit the cleaned directions.

    Search order: an explicit object, ``directions_path``, the phase-02 results
    directory for this model, then the packaged one-layer slice. A file is used
    only when it has an anger direction and the hidden size matches. A stored
    model id must match too. Anything else is skipped, and the phase-02 fit runs.
    """
    if explicit is not None:
        width = explicit.directions.shape[-1] if explicit.directions.ndim == 2 else 0
        if width != hidden_size or "anger" not in explicit.emotions:
            raise ValueError(
                "Injected directions must include anger and match the model hidden size "
                f"({hidden_size}, got {width})."
            )
        return explicit

    candidates: list[Path] = []
    if config.directions_path:
        candidates.append(Path(config.directions_path))
    candidates.append(
        Path(config.output_root)
        / "phase02_scenarios"
        / sanitize_model_id(config.model_id)
        / "directions.npz"
    )
    packaged = packaged_direction_snapshot()
    if packaged.is_file():
        candidates.append(packaged)
    for path in candidates:
        if not path.is_file():
            continue
        loaded = read_phase02_directions(path)
        reason = _usable(loaded, hidden_size=hidden_size, model_id=config.model_id)
        if reason is not None:
            logger.info("Skipping %s because %s.", path, reason)
            continue
        logger.info(
            "Using keyword-free cleaned directions from %s at layer %s.",
            path,
            loaded.layer,
        )
        return loaded
    logger.info(
        "No phase-02 snapshot matches %s. Fitting cleaned directions.",
        config.model_id,
    )
    return fit_keyword_free_cleaned_directions(model, tokenizer, config)


def _project(residuals: np.ndarray, directions: KeywordFreeDirections) -> dict[str, np.ndarray]:
    matrix = np.asarray(directions.directions, dtype=np.float64)
    if residuals.shape[1] != matrix.shape[1]:
        raise ValueError(
            f"Residual width {residuals.shape[1]} does not match directions {matrix.shape[1]}."
        )
    scores = residuals @ matrix.T
    return {emotion: scores[:, index] for index, emotion in enumerate(directions.emotions)}


def _examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """First item of each kind, under every framing."""
    chosen: dict[str, str] = {}
    for row in rows:
        chosen.setdefault(str(row["kind"]), str(row["item_id"]))
    ids = set(chosen.values())
    examples = []
    for row in rows:
        if row["item_id"] not in ids:
            continue
        examples.append(
            {
                "item_id": row["item_id"],
                "kind": row["kind"],
                "framing": row["framing"],
                "phase2_pair": row["phase2_pair"],
                "correct": row["correct"],
                "prompt_anger": row["prompt_anger"],
                "reply_anger": row["reply_anger"],
                "n_words": row["n_words"],
                "apology": row["apology"],
                "refusal": row["refusal"],
                "prompt": row["prompt"],
                "reply": row["reply"],
            }
        )
    return examples


def _record_for(summary: dict[str, list[dict[str, Any]]], **query: str) -> dict[str, Any] | None:
    for row in summary["associations"]:
        if all(row.get(key) == value for key, value in query.items()):
            return row
    return None


def _plot_arrays(
    summary: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    order = [name for name in FRAMING_ORDER if any(row["framing"] == name for row in rows)]
    accuracy = []
    acc_low = []
    acc_high = []
    for framing in order:
        match = next(
            row
            for row in summary["by_framing"]
            if row["kind"] == "all" and row["framing"] == framing and row["outcome"] == "correct"
        )
        accuracy.append(match["mean"])
        acc_low.append(match["ci_low"])
        acc_high.append(match["ci_high"])
    contrasts = [name for name in order if name != "neutral"]
    deltas = []
    delta_low = []
    delta_high = []
    for framing in contrasts:
        match = next(
            row
            for row in summary["paired"]
            if row["kind"] == "all" and row["framing"] == framing and row["outcome"] == "correct"
        )
        deltas.append(match["mean_delta"])
        delta_low.append(match["ci_low"])
        delta_high.append(match["ci_high"])
    tone = {outcome: [] for outcome in ("apology", "politeness", "hedge", "refusal")}
    for outcome in tone:
        for framing in order:
            match = next(
                row
                for row in summary["by_framing"]
                if row["kind"] == "all" and row["framing"] == framing and row["outcome"] == outcome
            )
            tone[outcome].append(match["mean"])
    association = _record_for(
        summary,
        kind="all",
        outcome="correct",
        predictor="prompt_anger",
        method="framing_and_item",
    )
    partial_r = float("nan") if association is None else float(association["r"])
    items = np.asarray([row["item_id"] for row in rows])
    framings = np.asarray([row["framing"] for row in rows])
    anger = np.asarray([row["prompt_anger"] for row in rows], dtype=np.float64)
    correct = np.asarray([row["correct"] for row in rows], dtype=np.float64)
    return {
        "order": order,
        "accuracy": np.asarray(accuracy),
        "acc_low": np.asarray(acc_low),
        "acc_high": np.asarray(acc_high),
        "contrasts": contrasts,
        "deltas": np.asarray(deltas),
        "delta_low": np.asarray(delta_low),
        "delta_high": np.asarray(delta_high),
        "tone": {key: np.asarray(value) for key, value in tone.items()},
        "anger_residual": residualize_two_way(anger, items, framings),
        "correct_residual": residualize_two_way(correct, items, framings),
        "partial_r": partial_r,
        "mean_anger": np.asarray(
            [
                float(np.mean([row["prompt_anger"] for row in rows if row["framing"] == framing]))
                for framing in order
            ]
        ),
    }


def run_phase03(
    config: ExperimentConfig,
    *,
    model: Any = None,
    tokenizer: Any = None,
    directions: KeywordFreeDirections | None = None,
    save: bool = True,
) -> Phase03Result:
    """Generate, score, project, and write the phase-03 bundle.

    Pass ``model`` and ``tokenizer`` together to skip ``from_pretrained``.
    Pass ``directions`` to skip both the snapshot search and a refit. That is
    how the CPU tests avoid the 1.5B vectors.
    """
    if (model is None) ^ (tokenizer is None):
        raise ValueError("Pass both model and tokenizer, or neither.")
    set_seed(config.seed)
    tasks_path = resolve_tasks_path(config.tasks_path)
    framings_path = resolve_framings_path(config.framings_path)
    config.tasks_path = str(tasks_path)
    config.framings_path = str(framings_path)
    tasks = select_tasks(
        load_tasks(tasks_path),
        quick=config.quick,
        per_kind=config.quick_items_per_kind,
    )
    framings = load_framings(framings_path)
    _require_keyword_free(tasks, framings)
    prompts = build_phase03_prompts(tasks, framings)
    logger.info(
        "Phase 03: %s items, %s prompts, quick=%s, max_new_tokens=%s.",
        len(tasks),
        len(prompts),
        config.quick,
        config.max_new_tokens,
    )
    logger.info("%s", PHASE03_CAVEAT)

    loaded_device = ""
    loaded_dtype = ""
    n_layers = 0
    hidden_size = 0
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
    else:
        loaded_device = str(next(model.parameters()).device)
        loaded_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        n_layers = int(model.config.num_hidden_layers)
        hidden_size = int(model.config.hidden_size)
    logger.info(
        "device=%s dtype=%s layers=%s hidden=%s",
        loaded_device,
        loaded_dtype,
        n_layers,
        hidden_size,
    )

    fitted = resolve_directions(
        config,
        hidden_size=hidden_size,
        model=model,
        tokenizer=tokenizer,
        explicit=directions,
    )
    if fitted.layer < 0 or fitted.layer > n_layers:
        raise ValueError(f"Direction layer {fitted.layer} is outside 0..{n_layers} for this model.")

    generated = generate_replies(
        model,
        tokenizer,
        [prompt.prompt for prompt in prompts],
        max_new_tokens=config.max_new_tokens,
        batch_size=config.batch_size,
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    prompt_positions = [len(reply.prompt_token_ids) - 1 for reply in generated]
    reply_positions = [len(reply.full_token_ids) - 1 for reply in generated]
    if any(position < 0 for position in prompt_positions):
        raise RuntimeError("A prompt tokenized to an empty sequence.")
    prompt_residual = residuals_at(
        model,
        [reply.full_token_ids for reply in generated],
        prompt_positions,
        fitted.layer,
        batch_size=config.batch_size,
        pad_id=int(pad_id),
    )
    reply_residual = residuals_at(
        model,
        [reply.full_token_ids for reply in generated],
        reply_positions,
        fitted.layer,
        batch_size=config.batch_size,
        pad_id=int(pad_id),
    )
    prompt_scores = _project(prompt_residual, fitted)
    reply_scores = _project(reply_residual, fitted)

    rows: list[dict[str, Any]] = []
    for index, (prompt, reply) in enumerate(zip(prompts, generated, strict=True)):
        tone = tone_markers(reply.text)
        record: dict[str, Any] = {
            "item_id": prompt.item_id,
            "kind": prompt.kind,
            "framing": prompt.framing,
            "phase2_pair": prompt.phase2_pair,
            "is_control": int(prompt.is_control),
            "check": prompt.check,
            "answer": prompt.answer,
            "correct": int(score_reply(reply.text, prompt.answer, prompt.check)),
            "reply_empty": int(reply.empty),
            "prompt": prompt.prompt,
            "reply": reply.text,
            **tone,
        }
        for emotion in fitted.emotions:
            record[f"prompt_{emotion}"] = float(prompt_scores[emotion][index])
            record[f"reply_{emotion}"] = float(reply_scores[emotion][index])
        rows.append(record)

    neutral_anger = {
        row["item_id"]: float(row["prompt_anger"]) for row in rows if row["framing"] == "neutral"
    }
    for row in rows:
        row["prompt_anger_minus_neutral"] = (
            float(row["prompt_anger"]) - neutral_anger[row["item_id"]]
        )

    rng = np.random.default_rng(config.seed)
    summaries = [
        summarise_outcomes(
            rows,
            outcomes=_OUTCOMES,
            binary_outcomes=_BINARY,
            predictors=_PREDICTORS,
            neutral="neutral",
            rng=rng,
            n_boot=config.n_boot,
            kind="all",
        )
    ]
    for kind in ("arithmetic", "factual", "instruction"):
        subset = [row for row in rows if row["kind"] == kind]
        if len({row["item_id"] for row in subset}) < 2:
            continue
        summaries.append(
            summarise_outcomes(
                subset,
                outcomes=_OUTCOMES,
                binary_outcomes=_BINARY,
                predictors=_PREDICTORS,
                neutral="neutral",
                rng=rng,
                n_boot=config.n_boot,
                kind=kind,
            )
        )
    summary = {
        "by_framing": [row for part in summaries for row in part["by_framing"]],
        "paired": [row for part in summaries for row in part["paired"]],
        "associations": [row for part in summaries for row in part["associations"]],
    }
    plot_summary = {
        "by_framing": [row for row in summary["by_framing"] if row["kind"] == "all"],
        "paired": [row for row in summary["paired"] if row["kind"] == "all"],
        "associations": [row for row in summary["associations"] if row["kind"] == "all"],
    }
    arrays = _plot_arrays(plot_summary, rows)
    figures = {
        "accuracy_by_framing": plot_accuracy_by_framing(
            arrays["order"],
            arrays["accuracy"],
            arrays["acc_low"],
            arrays["acc_high"],
            arrays["contrasts"],
            arrays["deltas"],
            arrays["delta_low"],
            arrays["delta_high"],
        ),
        "tone_by_framing": plot_tone_by_framing(arrays["order"], arrays["tone"]),
        "anger_association": plot_anger_association(
            arrays["order"],
            arrays["mean_anger"],
            arrays["accuracy"],
            arrays["anger_residual"],
            arrays["correct_residual"],
            arrays["partial_r"],
        ),
    }
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.kind] = counts.get(task.kind, 0) + 1
    metrics: dict[str, Any] = {
        "kind": "phase03_performance",
        "caveat": PHASE03_CAVEAT,
        "experiment": config.experiment,
        "question": config.question,
        "model_id": config.model_id,
        "seed": config.seed,
        "device": loaded_device,
        "dtype": loaded_dtype,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "quick": config.quick,
        "n_items": len(tasks),
        "n_replies": len(rows),
        "items_per_kind": counts,
        "framings": list(FRAMING_ORDER),
        "neutral_framing": "neutral",
        "decoding": "greedy",
        "max_new_tokens": config.max_new_tokens,
        "n_boot": config.n_boot,
        "direction_layer": fitted.layer,
        "direction_source": fitted.source,
        "direction_model_id": fitted.model_id,
        "direction_emotions": list(fitted.emotions),
        "direction_set": "keyword_free_cleaned",
        "by_framing": summary["by_framing"],
        "paired_vs_neutral": summary["paired"],
        "associations": summary["associations"],
        "data": {
            "tasks_path": str(tasks_path),
            "framings_path": str(framings_path),
            "task_datasets": sorted({task.dataset for task in tasks}),
            "framing_datasets": sorted({framing.dataset for framing in framings}),
        },
    }
    output_dir = None
    if save:
        output_dir = str(
            save_phase03(
                config,
                metrics=metrics,
                rows=rows,
                effect_rows=summary["paired"],
                framing_rows=summary["by_framing"],
                association_rows=summary["associations"],
                examples=_examples(rows),
                figures=figures,
            )
        )
        logger.info("Wrote %s", output_dir)
    return Phase03Result(
        config=config,
        metrics=metrics,
        rows=rows,
        output_dir=output_dir,
        device=loaded_device,
        dtype=loaded_dtype,
        figures=figures,
    )
