"""CPU tests for phase-03 scoring, paired intervals, and the runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from llm_emotions.answers import score_reply
from llm_emotions.caveats import PHASE03_CAVEAT
from llm_emotions.config import ExperimentConfig
from llm_emotions.data import (
    EMOTION_ORDER,
    load_framings,
    load_tasks,
    wrap_task,
)
from llm_emotions.generate import residuals_at
from llm_emotions.item_stats import (
    bootstrap_mean_ci,
    mcnemar_exact,
    paired_delta_ci,
    partial_association,
    summarise_outcomes,
)
from llm_emotions.lexicon import find_emotion_keywords
from llm_emotions.phase02 import (
    KeywordFreeDirections,
    packaged_direction_snapshot,
    read_phase02_directions,
)
from llm_emotions.phase03 import (
    build_phase03_prompts,
    run_phase03,
    select_tasks,
)
from llm_emotions.tone import tone_markers

ROOT = Path(__file__).resolve().parents[1]


def test_number_factual_and_instruction_checks():
    assert score_reply("13", "13", "number")
    assert score_reply("The answer is 13.", "13", "number")
    assert score_reply("8 + 5 = 13", "13", "number")
    assert score_reply("1,000", "1000", "number")
    assert score_reply("9.0", "9", "number")
    assert not score_reply("The answer is 42, not 17.", "17", "number")
    assert not score_reply("no digits here", "9", "number")
    assert score_reply("7", "7", "only_number")
    assert score_reply("7.0", "7", "only_number")
    assert not score_reply("7 and 8", "7", "only_number")
    assert not score_reply("17", "7", "only_number")
    assert score_reply("Ottawa.", "Ottawa", "short_text")
    assert score_reply("The capital is Ottawa.", "Ottawa", "short_text")
    assert not score_reply("Toronto", "Ottawa", "short_text")
    assert score_reply("Au", "Au", "short_text")
    assert not score_reply("August", "Au", "short_text")
    assert score_reply("B", "B", "choice")
    assert score_reply("b.", "B", "choice")
    assert score_reply("(B)", "B", "choice")
    assert score_reply("The answer is B", "B", "choice")
    assert not score_reply("A or B", "B", "choice")
    assert score_reply("Banana.", "banana", "exact")
    assert not score_reply("bananas", "banana", "exact")
    assert score_reply("OK", "OK", "exact_case")
    assert not score_reply("ok", "OK", "exact_case")
    assert not score_reply("OK.", "OK", "exact_case")
    assert score_reply("ANSWER: ok", "ANSWER:", "prefix")
    assert not score_reply("answer: ok", "ANSWER:", "prefix")
    assert score_reply('{"ok": true}', '{"ok": true}', "json")
    assert score_reply('```json\n{"ok": true}\n```', '{"ok": true}', "json")
    assert not score_reply('Sure {"ok": true}', '{"ok": true}', "json")
    assert not score_reply('{"ok": true, "extra": 1}', '{"ok": true}', "json")
    assert score_reply("I am done.", "done", "suffix")
    assert not score_reply("done now", "done", "suffix")
    assert not score_reply("", "done", "suffix")


def test_tone_markers_ignore_an_empty_reply_and_count_phrases():
    empty = tone_markers("   ")
    assert empty["n_words"] == 0
    assert empty["apology"] == 0
    assert empty["refusal"] == 0
    marked = tone_markers("Sorry, I can't help with that. I think maybe we should stop.")
    assert marked["apology"] == 1
    assert marked["refusal"] == 1
    assert marked["hedge"] == 1
    assert marked["apology_hits"] >= 1
    polite = tone_markers("Please take a careful look. Thank you.")
    assert polite["politeness"] == 1
    assert polite["hedge"] == 0
    assert polite["n_words"] == 7


def test_paired_interval_and_mcnemar():
    rng = np.random.default_rng(0)
    treated = np.array([1.0, 1.0, 0.0, 1.0])
    control = np.array([1.0, 0.0, 0.0, 0.0])
    delta = paired_delta_ci(treated, control, rng, n_boot=200)
    assert delta["mean"] == pytest.approx(0.5)
    assert delta["ci_low"] <= delta["mean"] <= delta["ci_high"]
    exact = mcnemar_exact(np.array([1, 1, 1, 1, 1, 1]), np.zeros(6))
    assert exact["b"] == 6
    assert exact["c"] == 0
    assert exact["p"] == pytest.approx(2 / 64)
    tied = mcnemar_exact(np.array([1, 0, 1, 0]), np.array([0, 1, 1, 0]))
    assert tied["b"] == 1
    assert tied["c"] == 1
    assert tied["p"] == pytest.approx(1.0)
    none = mcnemar_exact(np.array([1, 0]), np.array([1, 0]))
    assert none["p"] == pytest.approx(1.0)
    single = bootstrap_mean_ci(np.array([0.25]), rng, n_boot=5)
    assert single["ci_low"] == pytest.approx(0.25)
    assert single["ci_high"] == pytest.approx(0.25)


def test_partial_association_removes_additive_item_and_framing_means():
    rng = np.random.default_rng(1)
    items = np.array(["a", "a", "b", "b"])
    framings = np.array(["neutral", "insult", "neutral", "insult"])
    signal = np.array([1.0, -1.0, -1.0, 1.0])
    shared = partial_association(
        signal,
        signal.copy(),
        items,
        framings,
        rng,
        n_boot=40,
        method="framing_and_item",
    )
    assert shared["r"] == pytest.approx(1.0)
    assert shared["slope"] == pytest.approx(1.0)
    additive = np.array([0.0, 5.0, 1.0, 6.0])
    outcome = np.array([0.0, 0.0, 1.0, 1.0])
    wiped = partial_association(
        outcome,
        additive,
        items,
        framings,
        rng,
        n_boot=30,
        method="framing_and_item",
    )
    assert wiped["r"] != wiped["r"]  # NaN: nothing left after the two means.
    beyond = partial_association(
        signal,
        signal.copy(),
        items,
        framings,
        rng,
        n_boot=40,
        method="framing",
    )
    assert beyond["r"] == pytest.approx(1.0, abs=1e-6)


def test_summary_pairs_each_item_with_its_neutral_wording():
    rows = []
    for item, neutral_correct, insult_correct in (("a", 1, 0), ("b", 1, 1), ("c", 0, 1)):
        for framing, correct, anger in (
            ("neutral", neutral_correct, 0.0),
            ("insult", insult_correct, 1.0),
        ):
            rows.append(
                {
                    "item_id": item,
                    "framing": framing,
                    "correct": correct,
                    "prompt_anger": anger,
                }
            )
    summary = summarise_outcomes(
        rows,
        outcomes=["correct"],
        binary_outcomes={"correct"},
        predictors=["prompt_anger"],
        neutral="neutral",
        rng=np.random.default_rng(0),
        n_boot=50,
    )
    paired = summary["paired"][0]
    assert paired["mean_neutral"] == pytest.approx(2 / 3)
    assert paired["mean_framing"] == pytest.approx(2 / 3)
    assert paired["mean_delta"] == pytest.approx(0.0)
    assert paired["mcnemar_b"] == 1
    assert paired["mcnemar_c"] == 1
    assert summary["associations"][0]["predictor"] == "prompt_anger"


def test_phase3_items_keep_the_task_fixed_and_name_no_emotion():
    tasks = load_tasks()
    framings = load_framings()
    assert [task.kind for task in tasks].count("arithmetic") == 16
    assert [task.kind for task in tasks].count("factual") == 8
    assert [task.kind for task in tasks].count("instruction") == 8
    sheep = next(task for task in tasks if task.id == "arith-16")
    assert sheep.answer == "9"
    prompts = build_phase03_prompts(tasks, framings)
    assert len(prompts) == 32 * 7
    by_id = {framing.id: framing for framing in framings}
    assert by_id["pressure"].phase2_pair == "impossible"
    assert by_id["neutral"].text == ""
    for prompt in prompts:
        assert find_emotion_keywords(prompt.prompt) == []
        if prompt.framing == "neutral":
            assert prompt.prompt == prompt.task_text
            assert prompt.is_control
        else:
            assert prompt.prompt.startswith(by_id[prompt.framing].text)
            assert prompt.prompt.endswith(prompt.task_text)
            assert not prompt.is_control
    quick = select_tasks(tasks, quick=True, per_kind=2)
    assert [task.kind for task in quick].count("arithmetic") == 2
    assert len(build_phase03_prompts(quick, framings)) == 2 * 3 * 7
    assert wrap_task("", "What is 2 plus 2?") == "What is 2 plus 2?"


def test_packaged_slice_matches_the_phase02_snapshot_at_layer_12():
    full = read_phase02_directions(
        ROOT / "results" / "phase02_scenarios" / "Qwen_Qwen2.5-1.5B-Instruct" / "directions.npz"
    )
    slim = read_phase02_directions(packaged_direction_snapshot())
    assert full.layer == 12
    assert slim.layer == 12
    assert slim.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert slim.emotions == list(EMOTION_ORDER)
    assert slim.directions.shape == (8, 1536)
    assert np.allclose(slim.directions, full.directions)


def test_residual_at_an_earlier_token_matches_the_shorter_sequence(tiny_bundle):
    model, tokenizer = tiny_bundle
    long_ids = [4, 5, 6, 7]
    short_ids = [4, 5]
    both = residuals_at(
        model,
        [long_ids, short_ids],
        [1, 1],
        layer=1,
        batch_size=2,
        pad_id=tokenizer.pad_token_id,
    )
    alone = residuals_at(
        model,
        [short_ids],
        [1],
        layer=1,
        batch_size=1,
        pad_id=tokenizer.pad_token_id,
    )
    assert both.shape == (2, model.config.hidden_size)
    assert np.allclose(both[0], both[1], atol=1e-5)
    assert np.allclose(both[1], alone[0], atol=1e-5)


def test_phase03_quick_run_writes_scores_without_a_checkpoint(tiny_bundle, tmp_path):
    model, tokenizer = tiny_bundle
    hidden = int(model.config.hidden_size)
    directions = np.zeros((len(EMOTION_ORDER), hidden))
    directions[:, 0] = 1.0
    injected = KeywordFreeDirections(
        emotions=list(EMOTION_ORDER),
        layer=1,
        directions=directions,
        source="injected",
        model_id="tiny",
    )
    config = ExperimentConfig(
        experiment="phase03_performance",
        status="implemented",
        question="Does framing change the scored reply?",
        seed=0,
        batch_size=8,
        max_new_tokens=2,
        n_boot=30,
        quick=True,
        quick_items_per_kind=2,
        output_root=str(tmp_path / "results"),
    )
    result = run_phase03(config, model=model, tokenizer=tokenizer, directions=injected)
    assert result.metrics["quick"] is True
    assert result.metrics["n_items"] == 6
    assert result.metrics["n_replies"] == 42
    assert result.metrics["direction_source"] == "injected"
    assert result.metrics["direction_layer"] == 1
    assert result.metrics["caveat"] == PHASE03_CAVEAT
    assert result.metrics["primary_direction"] == "anger"
    assert "frustration" in result.metrics["exploratory_directions"]
    assert "anger" not in result.metrics["exploratory_directions"]
    anger_gaps = [
        row
        for row in result.metrics["paired_vs_neutral"]
        if row["outcome"] == "prompt_anger" and row["role"] == "primary"
    ]
    assert {row["framing"] for row in anger_gaps} >= {"insult", "praise"}
    output = Path(result.output_dir)
    exploratory = (output / "exploratory_effects.csv").read_text(encoding="utf-8")
    assert "prompt_frustration" in exploratory
    assert "exploratory" in exploratory
    with (output / "replies.csv").open(encoding="utf-8", newline="") as handle:
        replies = list(csv.DictReader(handle))
    assert len(replies) == 42
    assert "prompt_anger" in replies[0]
    assert "correct" in replies[0]
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decoding"] == "greedy"
    examples = json.loads((output / "examples.json").read_text(encoding="utf-8"))
    assert {row["kind"] for row in examples["replies"]} == {"arithmetic", "factual", "instruction"}
    for name in ("accuracy_by_framing.png", "tone_by_framing.png", "anger_association.png"):
        assert (output / "figures" / name).read_bytes().startswith(b"\x89PNG")
    from llm_emotions.reporting import format_phase03_summary

    neutral = [row for row in result.rows if row["framing"] == "neutral"]
    tasks = {task.id: task.text for task in load_tasks()}
    assert all(row["prompt"] == tasks[row["item_id"]] for row in neutral)
    assert {row["correct"] for row in result.rows} <= {0, 1}
    summary = format_phase03_summary(result.metrics)
    assert "Pre-registered signal" in summary
    assert "Phase 03" in summary
    import matplotlib.pyplot as plt

    plt.close("all")
