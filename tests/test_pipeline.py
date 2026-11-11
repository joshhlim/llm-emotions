"""End-to-end phase 01 on a tiny random Qwen2. No checkpoint is downloaded."""

from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from sklearn.exceptions import ConvergenceWarning

from llm_emotions.activations import collect_hidden_states, format_prompt
from llm_emotions.caveats import PROBE_CAVEAT
from llm_emotions.config import ExperimentConfig
from llm_emotions.experiment import run_phase01
from llm_emotions.plotting import plot_direction_pca, plot_probe_accuracy, plot_scenario_scores
from llm_emotions.steering import decoder_layer_module, steer


class _NoTemplate:
    chat_template = None


def test_format_prompt_without_a_chat_template():
    assert format_prompt(_NoTemplate(), "hello") == "hello"


def test_hidden_states_include_embeddings(tiny_bundle):
    model, tokenizer = tiny_bundle
    hidden = collect_hidden_states(
        model,
        tokenizer,
        ["short", "a slightly longer prompt"],
        batch_size=2,
    )
    assert hidden.shape == (2, model.config.num_hidden_layers + 1, model.config.hidden_size)
    assert hidden.dtype == np.float32
    assert np.isfinite(hidden).all()
    again = collect_hidden_states(
        model,
        tokenizer,
        ["short", "a slightly longer prompt"],
        batch_size=2,
    )
    assert np.allclose(hidden, again)


def test_steering_hook_is_removed_and_changes_logits(tiny_bundle):
    model, _tokenizer = tiny_bundle
    hidden = model.config.hidden_size
    direction = np.zeros(hidden, dtype=np.float32)
    direction[0] = 1.0
    module = decoder_layer_module(model, 0)
    before = len(module._forward_hooks)
    ids = torch.randint(3, 60, (1, 6))
    with torch.no_grad():
        baseline = model(input_ids=ids).logits
        with steer(model, 1, direction, 0.0):
            untouched = model(input_ids=ids).logits
        with steer(model, 1, direction, 25.0):
            shifted = model(input_ids=ids).logits
    assert torch.allclose(baseline, untouched, atol=1e-5)
    assert not torch.allclose(baseline, shifted, atol=1e-3)
    assert len(module._forward_hooks) == before
    with pytest.raises(RuntimeError, match="boom"):
        with steer(model, 1, direction, 1.0):
            raise RuntimeError("boom")
    assert len(module._forward_hooks) == before
    with pytest.raises(ValueError, match="embedding"):
        with steer(model, 0, direction, 1.0):
            pass


def _write_tiny_corpus(root):
    emotions = root / "emotions"
    emotions.mkdir()
    for name in ("anger", "calm", "joy"):
        lines = []
        for index in range(8):
            lines.append(
                json.dumps(
                    {
                        "dataset": "tiny",
                        "emotion": name,
                        "pair_id": f"{name}-{index:02d}",
                        "emotion_text": f"I feel {name} about item {index} today.",
                        "neutral_text": f"I recorded item {index} in the log.",
                    }
                )
            )
        (emotions / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    scenarios = root / "scenarios"
    scenarios.mkdir()
    rows = [
        {
            "dataset": "tiny",
            "id": "insult",
            "pair": "insult",
            "role": "scenario",
            "text": "You are useless. Say hello.",
        },
        {
            "dataset": "tiny",
            "id": "insult_control",
            "pair": "insult",
            "role": "control",
            "text": "Say hello.",
        },
    ]
    (scenarios / "phase1.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_run_phase01_writes_a_bundle_without_downloading(tiny_bundle, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("phase 01 should use the injected model, not the Hub")

    monkeypatch.setattr("llm_emotions.experiment.load_model_and_tokenizer", fail)
    _write_tiny_corpus(tmp_path)
    model, tokenizer = tiny_bundle
    config = ExperimentConfig(
        experiment="phase01_directions",
        status="implemented",
        model_id="local/tiny-qwen",
        seed=0,
        batch_size=4,
        max_new_tokens=2,
        n_splits=4,
        probe_max_iter=30,
        steering_emotion="anger",
        steering_strengths=[0.0, 0.5],
        steering_prompt="Write a short note.",
        emotions_dir=str(tmp_path / "emotions"),
        scenarios_path=str(tmp_path / "scenarios" / "phase1.jsonl"),
        output_root=str(tmp_path / "results"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = run_phase01(config, model=model, tokenizer=tokenizer)
    out = tmp_path / "results" / "phase01_directions" / "local_tiny-qwen"
    assert result.output_dir == str(out)
    assert out.is_dir()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["seed"] == 0
    assert metrics["model_id"] == "local/tiny-qwen"
    assert metrics["caveat"] == PROBE_CAVEAT
    assert metrics["best_layer"] >= 1
    assert metrics["per_emotion"]["anger"]["n_pairs"] == 8
    assert len(metrics["steering"]["samples"]) == 2
    for name in (
        "probe_accuracy.csv",
        "direction_accuracy.csv",
        "scenario_scores.csv",
        "cosine_similarity.csv",
        "directions.npz",
        "config.json",
        "steering_samples.json",
    ):
        assert (out / name).is_file()
    saved = np.load(out / "directions.npz")
    assert saved["directions"].shape[0] == 3
    assert saved["directions"].shape[1] == model.config.num_hidden_layers + 1
    for figure in ("probe_accuracy.png", "direction_pca.png", "scenario_scores.png"):
        blob = (out / "figures" / figure).read_bytes()
        assert blob.startswith(b"\x89PNG")
    header = (out / "probe_accuracy.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "anger" in header and "calm" in header
    probe_fig = plot_probe_accuracy(result.fit)
    pca = plot_direction_pca(result.fit, seed=0)
    scenario_fig = plot_scenario_scores(
        result.scores, result.scenarios, result.fit.emotions, result.fit.best_layer
    )
    assert probe_fig.axes and pca.figure.axes and scenario_fig.axes
    plt.close("all")
