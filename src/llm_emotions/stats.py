"""Uncertainty for a scenario-minus-control gap.

Two different questions get two different numbers.

* Across paraphrases: is the gap stable when the same situation is reworded?
  That is a one-sample mean, a bootstrap interval, Cohen's d, and an exact
  sign-flip p-value. With six paraphrases the smallest two-sided sign-flip
  p-value is 2/64.
* Across directions: is the gap large compared with a random axis of the same
  length? That is the matched-norm null. It says nothing about a population of
  prompts, and it is not corrected for the grid of emotions and situations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from llm_emotions.cleanup import unit_last


def matched_norm_directions(
    direction: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw Gaussian directions and rescale each one to ``||direction||``.

    The draw is uniform on the sphere only after normalisation. The rescale is
    what makes the null match the query direction's L2 norm, including a zero
    direction (every draw is then the zero vector).
    """
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1.")
    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    draws = rng.standard_normal((n_draws, vector.shape[0]))
    scaled = unit_last(draws) * float(np.linalg.norm(vector))
    return scaled


def random_direction_null(
    delta_residuals: np.ndarray,
    direction: np.ndarray,
    *,
    n_draws: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Compare the mean projection gap with matched-norm random directions.

    ``delta_residuals`` is ``(n_paraphrases, hidden)``: scenario residual minus
    the matched control residual, one row per paraphrase. The observed mean is
    the average over paraphrases of ``delta @ direction``.

    p-values use add-one smoothing, ``(count + 1) / (n_draws + 1)``, so a finite
    sample cannot report 0. ``null_p_greater`` is one-sided in the direction of
    a positive shift. ``null_p_two_sided`` uses the absolute mean.
    """
    deltas = np.asarray(delta_residuals, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    if deltas.ndim != 2 or deltas.shape[1] != vector.shape[0]:
        raise ValueError(
            "delta_residuals must have shape (n_paraphrases, hidden) matching direction."
        )
    if deltas.shape[0] < 1:
        raise ValueError("delta_residuals is empty.")
    observed = deltas @ vector
    observed_mean = float(observed.mean())
    draws = matched_norm_directions(vector, n_draws, rng)
    null_means = (deltas @ draws.T).mean(axis=0)
    greater = int(np.sum(null_means >= observed_mean))
    two_sided = int(np.sum(np.abs(null_means) >= abs(observed_mean)))
    total = int(n_draws + 1)
    return {
        "mean_delta": observed_mean,
        "null_p_greater": (greater + 1) / total,
        "null_p_two_sided": (two_sided + 1) / total,
        "null_mean": float(null_means.mean()),
        "null_sd": float(null_means.std(ddof=1)) if n_draws > 1 else float("nan"),
    }


def paraphrase_stats(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Mean gap across paraphrases, with a bootstrap interval and a sign-flip p.

    Cohen's d is ``mean / sample_sd`` across paraphrases. It describes agreement
    among wordings, not distance from the random-direction null. A zero sample
    standard deviation yields ``inf`` when the mean is nonzero, and 0 when the
    mean is also 0.

    The sign-flip test is exact for ``n <= 16`` (every assignment of signs).
    Above that it draws ``n_boot`` random sign patterns, always including the
    original. The p-value is the fraction of patterns whose absolute mean is at
    least the observed absolute mean.
    """
    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    n = int(values.shape[0])
    if n < 1:
        raise ValueError("deltas is empty.")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")
    mean = float(values.mean())
    if n < 2:
        sd = float("nan")
        cohens_d = float("nan")
        ci_low = mean
        ci_high = mean
    else:
        sd = float(values.std(ddof=1))
        if sd == 0.0:
            cohens_d = 0.0 if mean == 0.0 else float("inf")
        else:
            cohens_d = mean / sd
        index = rng.integers(0, n, size=(n_boot, n))
        boot_means = values[index].mean(axis=1)
        ci_low = float(np.quantile(boot_means, 0.025))
        ci_high = float(np.quantile(boot_means, 0.975))
    paraphrase_p = _sign_flip_p(values, mean, rng=rng, n_boot=n_boot)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "cohens_d": cohens_d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "paraphrase_p": paraphrase_p,
        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
    }


def _sign_flip_p(
    values: np.ndarray,
    mean: float,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> float:
    n = int(values.shape[0])
    threshold = abs(mean)
    if n <= 16:
        signs = _all_signs(n)
    else:
        bits = rng.integers(0, 2, size=(n_boot, n))
        signs = 1.0 - 2.0 * bits
        signs[0] = 1.0
    null_means = (signs * values).mean(axis=1)
    return float(np.mean(np.abs(null_means) >= threshold - 1e-12))


def _all_signs(n: int) -> np.ndarray:
    count = 1 << n
    index = np.arange(count, dtype=np.int64)[:, None]
    bits = (index >> np.arange(n, dtype=np.int64)) & 1
    return 1.0 - 2.0 * bits
