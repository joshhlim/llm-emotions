from __future__ import annotations

import json

import pytest

from llm_emotions.data import (
    EMOTION_ORDER,
    iter_scenario_controls,
    load_contrastive_pairs,
    load_scenarios,
    summarize_pairs,
)


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
