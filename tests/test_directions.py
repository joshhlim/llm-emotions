from __future__ import annotations

import numpy as np
import pytest

from llm_emotions.directions import fit_directions_and_probes, set_seed, unit


def _separable(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n, hidden, layers = 30, 8, 3
    signal = np.zeros(hidden)
    signal[0] = 1.0
    base = rng.normal(scale=0.3, size=(n, layers, hidden))
    positive = base + 3.0 * signal + rng.normal(scale=0.05, size=base.shape)
    negative = base - 3.0 * signal + rng.normal(scale=0.05, size=base.shape)
    return {"joy": {"emotion": positive, "neutral": negative}}


def test_unit_vector_and_zero():
    vector = unit(np.array([3.0, 4.0]))
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert np.allclose(unit(np.zeros(4)), 0.0)


def test_set_seed_pins_numpy():
    set_seed(123)
    first = np.random.rand(4)
    set_seed(123)
    second = np.random.rand(4)
    assert np.array_equal(first, second)


def test_probe_recovers_a_planted_direction():
    signal = np.zeros(8)
    signal[0] = 1.0
    fit = fit_directions_and_probes(_separable(), seed=0, n_splits=5, probe_max_iter=400)
    assert fit.best_layer == 1
    assert fit.probe_accuracy["joy"][1] == 1.0
    direction = fit.directions["joy"][2]
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert np.dot(direction, signal) > 0.95


def test_same_seed_reproduces_the_fit():
    data = _separable(2)
    first = fit_directions_and_probes(data, seed=0, n_splits=5, probe_max_iter=400)
    second = fit_directions_and_probes(data, seed=0, n_splits=5, probe_max_iter=400)
    assert np.allclose(first.probe_accuracy["joy"], second.probe_accuracy["joy"])
    assert np.allclose(first.directions["joy"], second.directions["joy"])


def test_best_layer_ignores_embeddings_and_unrequested_layers():
    rng = np.random.default_rng(1)
    n, hidden = 40, 6
    shared = rng.normal(size=(n, hidden))
    signal = np.zeros(hidden)
    signal[1] = 1.0
    positive = np.stack(
        [
            shared + rng.normal(scale=0.1, size=(n, hidden)),
            shared + rng.normal(scale=0.1, size=(n, hidden)),
            shared + 5.0 * signal,
        ],
        axis=1,
    )
    negative = np.stack(
        [
            shared + rng.normal(scale=0.1, size=(n, hidden)),
            shared + rng.normal(scale=0.1, size=(n, hidden)),
            shared - 5.0 * signal,
        ],
        axis=1,
    )
    data = {"calm": {"emotion": positive, "neutral": negative}}
    fit = fit_directions_and_probes(data, seed=0, n_splits=5, probe_max_iter=400)
    assert fit.best_layer == 2
    assert fit.probe_accuracy["calm"][2] == 1.0
    assert fit.probe_accuracy["calm"][2] > fit.probe_accuracy["calm"][0]

    only = fit_directions_and_probes(data, seed=0, n_splits=5, probe_max_iter=200, layers=[2])
    assert only.best_layer == 2
    assert np.isnan(only.probe_accuracy["calm"][0])


def test_too_few_pairs_for_the_requested_folds():
    data = {
        "anger": {
            "emotion": np.ones((2, 2, 4)),
            "neutral": np.zeros((2, 2, 4)),
        }
    }
    with pytest.raises(ValueError, match="need at least 4 pairs"):
        fit_directions_and_probes(data, seed=0, n_splits=4)
