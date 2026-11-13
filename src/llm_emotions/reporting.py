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


def format_phase02_summary(metrics: dict) -> str:
    """Short text form of the phase-02 metrics. Numbers come from the run, not from this file."""
    lines = [
        metrics["layer_selection"],
        (
            f"Chosen layer {metrics['best_layer']}. "
            f"Mean cleaned transfer accuracy there: "
            f"{metrics['mean_cleaned_transfer_at_best_layer']:.3f}."
        ),
    ]
    for dataset, stages in metrics["cosine_at_best_layer"].items():
        raw = stages["raw"]["mean_off_diagonal"]
        centered = stages["mean_centered"]["mean_off_diagonal"]
        cleaned = stages["cleaned"]["mean_off_diagonal"]
        lines.append(
            f"{dataset}: mean off-diagonal cosine "
            f"{raw:.3f} raw, {centered:.3f} mean-centred, {cleaned:.3f} cleaned."
        )
    lines.append("Pre-specified cells, keyword-free directions:")
    pre_specified = metrics.get("pre_specified") or []
    if not pre_specified:
        lines.append("  (none)")
    for row in pre_specified:
        lines.append(
            f"  {row['pair']} / {row['emotion']}: mean {row['mean_delta']:.3f}, "
            f"95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
            f"paraphrase p {row['paraphrase_p']:.3f}, "
            f"null p(greater) {row['null_p_greater']:.3f}."
        )
    lines.append(metrics["caveat"])
    return "\n".join(lines)


def _num(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def format_phase03_summary(metrics: dict) -> str:
    """Short text form of the phase-03 metrics. Numbers come from the run."""
    lines = [
        (
            f"{'Quick subset. ' if metrics.get('quick') else ''}"
            f"{metrics['n_items']} items, {metrics['n_replies']} replies, "
            f"layer {metrics['direction_layer']}, directions from {metrics['direction_source']}."
        ),
        (
            "Pre-registered signal: the prompt projection on "
            f"{metrics.get('primary_direction', 'anger')}. "
            "Other emotion directions are exploratory."
        ),
        "Anger projection minus the neutral wording of the same item:",
    ]
    for row in metrics["paired_vs_neutral"]:
        if row["kind"] != "all" or row["outcome"] != "prompt_anger":
            continue
        lines.append(
            f"  {row['framing']:12s} {row['mean_delta']:+.3f}  "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )
    lines.append("Accuracy by framing (95% bootstrap interval across items):")
    for row in metrics["by_framing"]:
        if row["kind"] != "all" or row["outcome"] != "correct":
            continue
        lines.append(
            f"  {row['framing']:12s} {row['mean']:.3f}  [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )
    lines.append("Accuracy minus the neutral wording of the same item:")
    for row in metrics["paired_vs_neutral"]:
        if row["kind"] != "all" or row["outcome"] != "correct":
            continue
        p_value = row["mcnemar_p"]
        p_text = "" if p_value == "" or p_value is None else f", McNemar p {float(p_value):.3f}"
        lines.append(
            f"  {row['framing']:12s} {row['mean_delta']:+.3f}  "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]{p_text}"
        )
    lines.append(
        "Partial association of the prompt's anger projection with correctness. "
        "framing removes the framing mean. framing_and_item also removes the item mean."
    )
    for row in metrics["associations"]:
        if row["kind"] != "all" or row["outcome"] != "correct":
            continue
        if row["predictor"] != "prompt_anger":
            continue
        lines.append(
            f"  {row['method']:18s} r {_num(row['r']):.3f}  "
            f"[{_num(row['r_ci_low']):.3f}, {_num(row['r_ci_high']):.3f}]  "
            f"slope {_num(row['slope']):.3f}"
        )
    lines.append(metrics["caveat"])
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
