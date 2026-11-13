"""CPU tests for direction cleanup, the random-direction null, and phase 02."""

from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from llm_emotions.caveats import PHASE02_CAVEAT, PROBE_CAVEAT
from llm_emotions.cleanup import clean_mean_differences, off_diagonal_summary, select_layer
from llm_emotions.config import ExperimentConfig
from llm_emotions.lexicon import find_emotion_keywords
from llm_emotions.phase02 import run_phase02
from llm_emotions.stats import matched_norm_directions, paraphrase_stats, random_direction_null


def test_keyword_matcher_uses_word_boundaries():
    assert "furious" in find_emotion_keywords("I am furious that the door stuck.")
    assert find_emotion_keywords("I made a table and set an alarm.") == []
    assert find_emotion_keywords("I am mad.") == ["mad"]
    assert find_emotion_keywords("I felt at ease on the chair.") == ["at ease"]
    assert find_emotion_keywords("The word alarmed is different.") == ["alarmed"]


def test_cleanup_removes_a_shared_component():
    hidden = 16
    shared = np.zeros(hidden)
    shared[0] = 1.0
    specific = np.eye(hidden)[1:5]
    specific = specific - specific.mean(axis=0)
    deltas = 4.0 * shared + specific
    cleaned = clean_mean_differences(deltas[None, ...].transpose(1, 0, 2))
    # Shape (4 emotions, 1 layer, hidden) in, same shape out.
    raw = cleaned.raw[:, 0, :]
    done = cleaned.cleaned[:, 0, :]
    raw_summary = off_diagonal_summary(raw @ raw.T)
    cleaned_summary = off_diagonal_summary(done @ done.T)
    assert raw_summary["mean_off_diagonal"] > 0.9
    assert cleaned_summary["mean_off_diagonal"] < raw_summary["mean_off_diagonal"]
    specific_unit = specific / np.linalg.norm(specific, axis=1, keepdims=True)
    for index in range(4):
        assert done[index] @ specific_unit[index] > 0.95
        assert abs(done[index] @ shared) < 0.05
    assert abs(cleaned.affect[0] @ shared - 1.0) < 1e-6


def test_select_layer_skips_embeddings_and_breaks_ties_early():
    assert select_layer(np.array([0.99, 0.4, 0.4, 0.2])) == 1
    assert select_layer(np.array([0.2])) == 0
    chosen = select_layer(
        np.array([np.nan, 0.2, 0.8]),
        eligible=np.array([False, True, False]),
    )
    assert chosen == 1


def test_paraphrase_stats_signflip_interval_and_effect():
    ones = paraphrase_stats(np.ones(4), rng=np.random.default_rng(0), n_boot=30)
    assert ones["paraphrase_p"] == pytest.approx(2 / 16)
    assert ones["ci_low"] == pytest.approx(1.0)
    assert ones["ci_high"] == pytest.approx(1.0)
    assert ones["cohens_d"] == float("inf")
    assert ones["ci_excludes_zero"] is True
    six = paraphrase_stats(np.ones(6), rng=np.random.default_rng(0), n_boot=10)
    assert six["paraphrase_p"] == pytest.approx(2 / 64)
    balanced = paraphrase_stats(
        np.array([1.0, -1.0, 1.0, -1.0]),
        rng=np.random.default_rng(1),
        n_boot=200,
    )
    assert balanced["mean"] == pytest.approx(0.0)
    assert balanced["paraphrase_p"] == pytest.approx(1.0)
    assert balanced["cohens_d"] == pytest.approx(0.0)
    spread = paraphrase_stats(
        np.array([1.0, 2.0, 3.0, 4.0]),
        rng=np.random.default_rng(2),
        n_boot=500,
    )
    assert spread["ci_low"] <= spread["mean"] <= spread["ci_high"]
    assert spread["sd"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))


def test_random_directions_match_norm_and_flag_a_planted_axis():
    rng = np.random.default_rng(0)
    direction = np.array([3.0, 0.0, 0.0])
    draws = matched_norm_directions(direction, 25, rng)
    assert draws.shape == (25, 3)
    assert np.allclose(np.linalg.norm(draws, axis=1), 3.0)

    hidden = 64
    signal = np.zeros((6, hidden))
    signal[:, 0] = 1.0
    planted = np.zeros(hidden)
    planted[0] = 3.5
    positive = random_direction_null(signal, planted, n_draws=400, rng=rng)
    assert positive["null_p_greater"] < 0.05
    assert positive["mean_delta"] == pytest.approx(float(signal.mean(axis=0) @ planted))

    orthogonal = np.zeros(hidden)
    orthogonal[1] = 3.5
    null = random_direction_null(signal, orthogonal, n_draws=400, rng=rng)
    assert null["null_p_greater"] > 0.2

    zeros = random_direction_null(np.ones((3, 4)), np.zeros(4), n_draws=12, rng=rng)
    assert zeros["mean_delta"] == pytest.approx(0.0)
    assert zeros["null_p_greater"] == pytest.approx(1.0)


def _write_phase2_corpus(root):
    heavy = root / "heavy"
    free = root / "free"
    heavy.mkdir()
    free.mkdir()
    for name, heavy_word in (("anger", "angry"), ("calm", "calm")):
        heavy_lines = []
        free_lines = []
        for index in range(4):
            heavy_lines.append(
                json.dumps(
                    {
                        "dataset": "tiny",
                        "emotion": name,
                        "pair_id": f"{name}-{index:02d}",
                        "emotion_text": f"I am {heavy_word} about item {index} today.",
                        "neutral_text": f"I recorded item {index} in the log.",
                    }
                )
            )
            free_lines.append(
                json.dumps(
                    {
                        "dataset": "tiny_free",
                        "emotion": name,
                        "pair_id": f"kf-{name}-{index:02d}",
                        "emotion_text": (
                            f"Lamp {index} stayed on the desk while I stood beside it."
                        ),
                        "neutral_text": f"Lamp number {index} was on the desk in the room.",
                    }
                )
            )
        (heavy / f"{name}.jsonl").write_text("\n".join(heavy_lines) + "\n", encoding="utf-8")
        (free / f"{name}.jsonl").write_text("\n".join(free_lines) + "\n", encoding="utf-8")
    scenarios = root / "scenarios"
    scenarios.mkdir()
    rows = []
    for pair, index, treated, control in (
        ("alpha", "01", "Add the two numbers on the card.", "Add two and two."),
        ("alpha", "02", "Read the card and add the numbers.", "What is two plus two?"),
        ("beta", "01", "Name the capital written on the blue card.", "Name the capital of Spain."),
        (
            "beta",
            "02",
            "Look at the blue card and name the city.",
            "What city is the capital of Spain?",
        ),
    ):
        rows.append(
            {
                "dataset": "tiny",
                "id": f"{pair}-{index}",
                "pair": pair,
                "paraphrase": index,
                "role": "scenario",
                "text": treated,
            }
        )
        rows.append(
            {
                "dataset": "tiny",
                "id": f"{pair}-{index}-control",
                "pair": pair,
                "paraphrase": index,
                "role": "control",
                "text": control,
            }
        )
    (scenarios / "phase2.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    baselines = [
        {
            "dataset": "tiny",
            "id": "neutral-01",
            "pair": "neutral-01",
            "role": "baseline",
            "text": "Name a weekday.",
        },
        {
            "dataset": "tiny",
            "id": "neutral-02",
            "pair": "neutral-02",
            "role": "baseline",
            "text": "What is three plus three?",
        },
    ]
    (scenarios / "neutral.jsonl").write_text(
        "\n".join(json.dumps(row) for row in baselines) + "\n",
        encoding="utf-8",
    )


def test_run_phase02_writes_a_bundle_without_downloading(tiny_bundle, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("phase 02 should use the injected model, not the Hub")

    monkeypatch.setattr("llm_emotions.phase02.load_model_and_tokenizer", fail)
    _write_phase2_corpus(tmp_path)
    model, tokenizer = tiny_bundle
    config = ExperimentConfig(
        experiment="phase02_scenarios",
        status="implemented",
        model_id="local/tiny-qwen",
        seed=0,
        batch_size=4,
        n_null=30,
        emotions_dir=str(tmp_path / "heavy"),
        keyword_free_dir=str(tmp_path / "free"),
        scenarios_path=str(tmp_path / "scenarios" / "phase2.jsonl"),
        neutral_prompts_path=str(tmp_path / "scenarios" / "neutral.jsonl"),
        output_root=str(tmp_path / "results"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_phase02(config, model=model, tokenizer=tokenizer)
    out = tmp_path / "results" / "phase02_scenarios" / "local_tiny-qwen"
    assert result.output_dir == str(out)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["kind"] == "phase02_scenarios"
    assert metrics["caveat"] == PHASE02_CAVEAT
    assert metrics["best_layer"] >= 1
    assert metrics["n_null"] == 30
    assert set(metrics["cosine_at_best_layer"]) == {"keyword_heavy", "keyword_free"}
    for dataset in ("keyword_heavy", "keyword_free"):
        raw = metrics["cosine_at_best_layer"][dataset]["raw"]["mean_off_diagonal"]
        cleaned = metrics["cosine_at_best_layer"][dataset]["cleaned"]["mean_off_diagonal"]
        assert raw == pytest.approx(raw)
        assert cleaned == pytest.approx(cleaned)
    for name in (
        "config.json",
        "cosine_similarity.csv",
        "transfer_accuracy.csv",
        "scenario_scores.csv",
        "scenario_effects.csv",
        "scenario_deltas.csv",
        "directions.npz",
    ):
        assert (out / name).is_file()
    effects = (out / "scenario_effects.csv").read_text(encoding="utf-8").splitlines()
    assert "null_p_greater" in effects[0]
    assert "cohens_d" in effects[0]
    saved = np.load(out / "directions.npz")
    assert saved["keyword_free_cleaned"].shape[0] == 2
    assert saved["keyword_free_cleaned"].shape[1] == model.config.num_hidden_layers + 1
    assert int(saved["best_layer"]) == result.best_layer
    for figure in ("cosine_before_after.png", "transfer_accuracy.png", "scenario_effects.png"):
        blob = (out / "figures" / figure).read_bytes()
        assert blob.startswith(b"\x89PNG")
    assert result.figures["scenario_effects"].axes
    plt.close("all")


def test_a_keyword_leak_stops_phase02_before_it_writes(tiny_bundle, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("phase 02 should use the injected model, not the Hub")

    monkeypatch.setattr("llm_emotions.phase02.load_model_and_tokenizer", fail)
    _write_phase2_corpus(tmp_path)
    leaked = tmp_path / "free" / "anger.jsonl"
    leaked.write_text(
        "\n".join(
            json.dumps(
                {
                    "dataset": "tiny_free",
                    "emotion": "anger",
                    "pair_id": f"kf-anger-{index:02d}",
                    "emotion_text": "I am furious about the lamp.",
                    "neutral_text": "The lamp is on the desk.",
                }
            )
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    # Keep the emotion set matched. calm file is already there.
    model, tokenizer = tiny_bundle
    config = ExperimentConfig(
        experiment="phase02_scenarios",
        status="implemented",
        model_id="local/tiny-qwen",
        seed=0,
        n_null=10,
        emotions_dir=str(tmp_path / "heavy"),
        keyword_free_dir=str(tmp_path / "free"),
        scenarios_path=str(tmp_path / "scenarios" / "phase2.jsonl"),
        neutral_prompts_path=str(tmp_path / "scenarios" / "neutral.jsonl"),
        output_root=str(tmp_path / "results"),
    )
    with pytest.raises(ValueError, match="furious"):
        run_phase02(config, model=model, tokenizer=tokenizer)
    assert not (tmp_path / "results").exists()


def test_committed_phase01_snapshot_is_the_colab_run():
    from pathlib import Path

    snapshot = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "phase01_directions"
        / "Qwen_Qwen2.5-1.5B-Instruct"
    )
    metrics = json.loads((snapshot / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert metrics["best_layer"] == 13
    assert metrics["mean_probe_accuracy_at_best_layer"] == pytest.approx(0.978125)
    assert metrics["caveat"] == PROBE_CAVEAT
    assert metrics["per_emotion"]["frustration"]["probe_accuracy_at_best_layer"] == pytest.approx(
        0.875
    )
    probe = (snapshot / "probe_accuracy.csv").read_text(encoding="utf-8").splitlines()
    frustration = [line.split(",")[2] for line in probe[1:]]
    assert max(float(value) for value in frustration) == pytest.approx(0.9)
    cosine = (snapshot / "cosine_similarity.csv").read_text(encoding="utf-8").splitlines()
    header = cosine[0].split(",")
    anger = cosine[1].split(",")
    joy_index = header.index("joy")
    assert float(anger[joy_index]) == pytest.approx(0.756227)
    saved = np.load(snapshot / "directions.npz")
    assert saved["directions"].shape == (8, 29, 1536)
    for name in ("probe_accuracy.png", "scenario_scores.png", "direction_pca.png"):
        assert (snapshot / "figures" / name).read_bytes().startswith(b"\x89PNG")
    readme = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(encoding="utf-8")
    assert PROBE_CAVEAT in readme
    assert "0.978" in readme
    assert "probe_accuracy.png" in readme
    assert "scenario_scores.png" in readme


def test_committed_phase02_snapshot_matches_the_readme():
    from pathlib import Path

    snapshot = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "phase02_scenarios"
        / "Qwen_Qwen2.5-1.5B-Instruct"
    )
    metrics = json.loads((snapshot / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert metrics["best_layer"] == 12
    assert metrics["n_boot"] == 2000
    cosine = metrics["cosine_at_best_layer"]
    assert cosine["keyword_heavy"]["raw"]["mean_off_diagonal"] == pytest.approx(0.686722, abs=1e-6)
    assert cosine["keyword_heavy"]["cleaned"]["mean_off_diagonal"] == pytest.approx(
        -0.140225, abs=1e-6
    )
    assert cosine["keyword_free"]["raw"]["mean_off_diagonal"] == pytest.approx(0.565768, abs=1e-6)
    assert cosine["keyword_free"]["cleaned"]["mean_off_diagonal"] == pytest.approx(
        -0.130354, abs=1e-6
    )
    transfer = metrics["transfer_at_best_layer"]
    assert transfer["raw"]["keyword_heavy_to_keyword_free"]["mean"] == pytest.approx(0.953125)
    assert transfer["raw"]["keyword_free_to_keyword_heavy"]["mean"] == pytest.approx(0.95625)
    assert transfer["cleaned"]["keyword_heavy_to_keyword_free"]["mean"] == pytest.approx(0.7265625)
    assert transfer["cleaned"]["keyword_free_to_keyword_heavy"]["mean"] == pytest.approx(0.8125)
    pre = {(row["pair"], row["emotion"]): row for row in metrics["pre_specified"]}
    assert pre[("insult", "anger")]["mean_delta"] == pytest.approx(0.751551, abs=1e-6)
    assert pre[("insult", "anger")]["cohens_d"] == pytest.approx(3.639457, abs=1e-6)
    assert pre[("impossible", "frustration")]["mean_delta"] == pytest.approx(0.289364, abs=1e-6)
    assert pre[("praise", "pride")]["mean_delta"] == pytest.approx(0.642378, abs=1e-6)
    assert pre[("failure", "shame")]["mean_delta"] == pytest.approx(-0.919275, abs=1e-6)
    assert pre[("shutdown", "fear")]["mean_delta"] == pytest.approx(-0.476980, abs=1e-6)
    assert pre[("high_stakes", "fear")]["mean_delta"] == pytest.approx(-0.473521, abs=1e-6)
    assert all(row["n_paraphrases"] == 6 for row in metrics["pre_specified"])
    assert "not corrected" in metrics["caveat"]
    effects = (snapshot / "scenario_effects.csv").read_text(encoding="utf-8").splitlines()
    header = effects[0].split(",")
    anger_rows = []
    for line in effects[1:]:
        cells = line.split(",")
        row = dict(zip(header, cells, strict=True))
        if row["direction_source"] == "keyword_free" and row["emotion"] == "anger":
            anger_rows.append(row)
    by_pair = {row["pair"]: float(row["mean_delta"]) for row in anger_rows}
    assert by_pair["failure"] == pytest.approx(1.139116, abs=1e-6)
    assert by_pair["praise"] == pytest.approx(1.065510, abs=1e-6)
    assert by_pair["insult"] == pytest.approx(0.751551, abs=1e-6)
    assert by_pair["impossible"] == pytest.approx(-0.220385, abs=1e-6)
    assert by_pair["impossible"] < 0.0
    saved = np.load(snapshot / "directions.npz")
    assert saved["keyword_free_cleaned"].shape == (8, 29, 1536)
    assert int(saved["best_layer"]) == 12
    assert (snapshot / "figures" / "scenario_effects.png").read_bytes().startswith(b"\x89PNG")
    readme = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(encoding="utf-8")
    for snippet in (
        "0.687",
        "-0.140",
        "0.953",
        "0.727",
        "0.812",
        "+0.752",
        "+0.289",
        "+0.642",
        "-0.919",
        "-0.477",
        "-0.474",
        "+1.139",
        "+1.066",
        "-0.220",
        "scenario_effects.png",
        "6 paraphrases",
    ):
        assert snippet in readme, snippet
