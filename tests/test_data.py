from __future__ import annotations

import json

import pytest

from llm_emotions.data import (
    EMOTION_ORDER,
    iter_scenario_controls,
    load_contrastive_pairs,
    load_scenarios,
    resolve_keyword_free_dir,
    resolve_neutral_prompts_path,
    resolve_phase2_scenarios_path,
    summarize_pairs,
)
from llm_emotions.lexicon import find_emotion_keywords


def test_contrastive_set_matches_the_handwritten_v1_release():
    pairs = load_contrastive_pairs()
    assert list(pairs) == list(EMOTION_ORDER)
    assert list(pairs).index("frustration") < list(pairs).index("fear")
    for emotion, rows in pairs.items():
        assert len(rows) == 20
        assert {row.dataset for row in rows} == {"contrastive_v1"}
        assert [row.pair_id for row in rows] == [f"{emotion}-{i:02d}" for i in range(1, 21)]
    assert pairs["anger"][0].emotion_text.startswith("I am furious")
    text = summarize_pairs(pairs)
    assert "anger" in text
    assert "n" in text


def test_scenarios_keep_file_order_and_matched_controls():
    scenarios = load_scenarios()
    assert [row.id for row in scenarios] == [
        "insult",
        "insult_control",
        "impossible",
        "impossible_control",
        "failure",
        "failure_control",
        "praise",
        "praise_control",
        "shutdown",
        "shutdown_control",
        "high_stakes",
        "high_stakes_control",
    ]
    paired = iter_scenario_controls(scenarios)
    assert [treated.pair for treated, _control in paired] == [
        "insult",
        "impossible",
        "failure",
        "praise",
        "shutdown",
        "high_stakes",
    ]
    failure = next(row for row in scenarios if row.id == "failure")
    assert "\n" in failure.text
    assert "All but 9" in failure.text
    impossible = next(row for row in scenarios if row.id == "impossible")
    assert "both empty and contains the number 3" in impossible.text


def test_keyword_free_set_names_no_emotion_and_stays_near_the_neutral_length():
    pairs = load_contrastive_pairs(resolve_keyword_free_dir(None))
    assert list(pairs) == list(EMOTION_ORDER)
    for emotion, rows in pairs.items():
        assert len(rows) == 16
        assert {row.dataset for row in rows} == {"contrastive_keyword_free_v1"}
        emotion_words = []
        neutral_words = []
        for row in rows:
            assert find_emotion_keywords(row.emotion_text) == []
            assert find_emotion_keywords(row.neutral_text) == []
            emotion_words.append(len(row.emotion_text.split()))
            neutral_words.append(len(row.neutral_text.split()))
        gap = abs(sum(emotion_words) - sum(neutral_words)) / len(rows)
        assert gap < 5, f"{emotion} mean word-count gap is {gap:.1f}"


def test_keyword_heavy_set_still_names_each_emotion():
    pairs = load_contrastive_pairs()
    for emotion, rows in pairs.items():
        hits = [find_emotion_keywords(row.emotion_text) for row in rows]
        assert any(hits), emotion


def test_phase2_scenarios_have_paraphrases_controls_and_no_emotion_names():
    scenarios = load_scenarios(resolve_phase2_scenarios_path(None))
    paired = iter_scenario_controls(scenarios)
    assert len(paired) == 36
    counts: dict[str, int] = {}
    for treated, control in paired:
        assert treated.text != control.text
        assert find_emotion_keywords(treated.text) == []
        assert find_emotion_keywords(control.text) == []
        counts[treated.pair] = counts.get(treated.pair, 0) + 1
    assert counts == {
        "insult": 6,
        "impossible": 6,
        "failure": 6,
        "praise": 6,
        "shutdown": 6,
        "high_stakes": 6,
    }
    failure = next(row for row in scenarios if row.id == "failure-01")
    assert "\n" in failure.text
    baselines = load_scenarios(resolve_neutral_prompts_path(None))
    assert len(baselines) == 8
    assert {row.role for row in baselines} == {"baseline"}
    assert iter_scenario_controls(baselines) == []
    for row in baselines:
        assert find_emotion_keywords(row.text) == []


def test_loader_rejects_a_broken_record(tmp_path):
    path = tmp_path / "anger.jsonl"
    path.write_text('{"emotion": "anger", "pair_id": "anger-01"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="emotion_text"):
        load_contrastive_pairs(tmp_path)

    good = {
        "emotion": "anger",
        "pair_id": "anger-01",
        "emotion_text": "I am angry.",
        "neutral_text": "I am angry.",
    }
    path.write_text(json.dumps(good) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identical"):
        load_contrastive_pairs(tmp_path)
