"""Item-level uncertainty for phase 03.

Framing contrasts resample items. The neutral wording of the same item is the
paired control. Two partial associations are reported for a projection:

* residualizing on the framing label, which is the "beyond the framing" question
* residualizing on the framing and the item, which also removes task difficulty

Neither association is a steering effect. The bootstrap is a percentile
interval. It is not corrected for the number of framings or outcomes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boot: int = 2000,
) -> dict[str, float]:
    """Mean and a 95% percentile bootstrap interval, resampling rows."""
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size < 1:
        raise ValueError("values is empty.")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")
    mean = float(sample.mean())
    if sample.size == 1:
        return {"n": 1.0, "mean": mean, "ci_low": mean, "ci_high": mean}
    index = rng.integers(0, sample.size, size=(n_boot, sample.size))
    means = sample[index].mean(axis=1)
    return {
        "n": float(sample.size),
        "mean": mean,
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def paired_delta_ci(
    treated: np.ndarray,
    control: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boot: int = 2000,
) -> dict[str, float]:
    """Mean of treated minus control, with a bootstrap interval over items."""
    left = np.asarray(treated, dtype=np.float64).reshape(-1)
    right = np.asarray(control, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("treated and control must have the same shape.")
    if left.size < 1:
        raise ValueError("treated is empty.")
    stats = bootstrap_mean_ci(left - right, rng, n_boot=n_boot)
    stats["mean_treated"] = float(left.mean())
    stats["mean_control"] = float(right.mean())
    return stats


def mcnemar_exact(treated: np.ndarray, control: np.ndarray) -> dict[str, float]:
    """Exact two-sided McNemar test for paired 0/1 outcomes.

    ``b`` is treated-only successes. ``c`` is control-only successes. The
    p-value is the two-sided binomial probability of a split at least this
    uneven under a fair coin, capped at 1. No discordant pairs gives p = 1.
    """
    left = np.asarray(treated).reshape(-1)
    right = np.asarray(control).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("treated and control must have the same shape.")
    if left.size < 1:
        raise ValueError("treated is empty.")
    if not np.isin(left, (0, 1)).all() or not np.isin(right, (0, 1)).all():
        raise ValueError("McNemar inputs must be 0/1.")
    only_treated = int(np.sum((left == 1) & (right == 0)))
    only_control = int(np.sum((left == 0) & (right == 1)))
    return {
        "b": float(only_treated),
        "c": float(only_control),
        "n_discordant": float(only_treated + only_control),
        "p": _binomial_two_sided(only_treated, only_control),
    }


def _binomial_two_sided(successes: int, failures: int) -> float:
    n = successes + failures
    if n == 0:
        return 1.0
    smaller = min(successes, failures)
    if smaller >= n - smaller:
        return 1.0
    ways = sum(math.comb(n, index) for index in range(smaller + 1))
    return float(min(1.0, 2.0 * ways / (1 << n)))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _slope(predictor: np.ndarray, outcome: np.ndarray) -> float:
    """OLS slope of outcome on predictor. Both arrays should already be residualized."""
    x = np.asarray(predictor, dtype=np.float64).reshape(-1)
    y = np.asarray(outcome, dtype=np.float64).reshape(-1)
    xx = float(np.sum(x * x))
    if xx < 1e-12:
        return float("nan")
    return float(np.sum(x * y) / xx)


def residualize_groups(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean. The residual is the within-group deviation."""
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = np.asarray(groups)
    if sample.shape[0] != labels.shape[0]:
        raise ValueError("values and groups must have the same length.")
    residuals = np.empty_like(sample)
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        residuals[mask] = sample[mask] - sample[mask].mean()
    return residuals


def residualize_two_way(
    values: np.ndarray,
    row_groups: np.ndarray,
    col_groups: np.ndarray,
) -> np.ndarray:
    """Remove additive row and column means.

    The design this is used for is balanced: every item appears under every
    framing. The residual is ``value - item mean - framing mean + grand mean``.
    """
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    rows = np.asarray(row_groups)
    cols = np.asarray(col_groups)
    if not (sample.shape[0] == rows.shape[0] == cols.shape[0]):
        raise ValueError("values, row groups, and column groups must share a length.")
    if sample.size < 1:
        raise ValueError("values is empty.")
    grand = float(sample.mean())
    row_mean = {key: float(sample[rows == key].mean()) for key in dict.fromkeys(rows.tolist())}
    col_mean = {key: float(sample[cols == key].mean()) for key in dict.fromkeys(cols.tolist())}
    residuals = np.empty_like(sample)
    for index, (row, col, value) in enumerate(zip(rows, cols, sample, strict=True)):
        residuals[index] = value - row_mean[row] - col_mean[col] + grand
    return residuals


def _finite_interval(samples: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in samples if np.isfinite(value)], dtype=np.float64)
    if finite.size < 10:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _partial_point(
    outcome: np.ndarray,
    predictor: np.ndarray,
    *,
    items: np.ndarray,
    framings: np.ndarray,
    method: str,
) -> tuple[float, float]:
    if method == "framing":
        y = residualize_groups(outcome, framings)
        x = residualize_groups(predictor, framings)
    elif method == "framing_and_item":
        y = residualize_two_way(outcome, items, framings)
        x = residualize_two_way(predictor, items, framings)
    else:
        raise ValueError(f"Unknown partial method {method!r}.")
    return _pearson(x, y), _slope(x, y)


def partial_association(
    outcome: np.ndarray,
    predictor: np.ndarray,
    items: np.ndarray,
    framings: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boot: int = 2000,
    method: str = "framing",
) -> dict[str, Any]:
    """Pearson r and OLS slope after residualizing, with an item bootstrap.

    Resampling draws items with replacement and keeps every framing of a drawn
    item. The residual means are recomputed inside each draw.
    """
    y = np.asarray(outcome, dtype=np.float64).reshape(-1)
    x = np.asarray(predictor, dtype=np.float64).reshape(-1)
    item_ids = np.asarray(items)
    framing_ids = np.asarray(framings)
    if not (y.shape[0] == x.shape[0] == item_ids.shape[0] == framing_ids.shape[0]):
        raise ValueError("outcome, predictor, items, and framings must share a length.")
    if y.size < 1:
        raise ValueError("outcome is empty.")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")
    point_r, point_slope = _partial_point(y, x, items=item_ids, framings=framing_ids, method=method)
    order = list(dict.fromkeys(item_ids.tolist()))
    grouped = {item: np.flatnonzero(item_ids == item) for item in order}
    boot_r: list[float] = []
    boot_slope: list[float] = []
    n_items = len(order)
    draws = rng.integers(0, n_items, size=(n_boot, n_items))
    for draw in draws:
        chosen = np.concatenate([grouped[order[int(index)]] for index in draw])
        r_value, slope = _partial_point(
            y[chosen],
            x[chosen],
            items=item_ids[chosen],
            framings=framing_ids[chosen],
            method=method,
        )
        boot_r.append(r_value)
        boot_slope.append(slope)
    r_low, r_high = _finite_interval(boot_r)
    slope_low, slope_high = _finite_interval(boot_slope)
    return {
        "method": method,
        "n": int(y.size),
        "n_items": n_items,
        "r": point_r,
        "r_ci_low": r_low,
        "r_ci_high": r_high,
        "slope": point_slope,
        "slope_ci_low": slope_low,
        "slope_ci_high": slope_high,
    }


def _aligned(rows: list[dict[str, Any]], framing: str, outcome: str) -> dict[str, float]:
    aligned: dict[str, float] = {}
    for row in rows:
        if row["framing"] != framing:
            continue
        item_id = str(row["item_id"])
        if item_id in aligned:
            raise ValueError(f"Item {item_id!r} has two rows for framing {framing!r}.")
        aligned[item_id] = float(row[outcome])
    return aligned


def summarise_outcomes(
    rows: list[dict[str, Any]],
    *,
    outcomes: list[str],
    binary_outcomes: set[str],
    predictors: list[str],
    neutral: str,
    rng: np.random.Generator,
    n_boot: int = 2000,
    kind: str = "all",
) -> dict[str, list[dict[str, Any]]]:
    """Framing means, paired gaps versus neutral, and partial associations.

    ``rows`` are one reply each. Every item must appear under every framing,
    including neutral. ``kind`` is copied onto the records so a caller can stack
    an overall table and per-kind tables.
    """
    if not rows:
        raise ValueError("rows is empty.")
    framings = list(dict.fromkeys(str(row["framing"]) for row in rows))
    if neutral not in framings:
        raise ValueError(f"Neutral framing {neutral!r} is missing.")
    item_order = list(dict.fromkeys(str(row["item_id"]) for row in rows))
    by_framing: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for outcome in outcomes:
        neutral_map = _aligned(rows, neutral, outcome)
        if set(neutral_map) != set(item_order):
            raise ValueError(f"Outcome {outcome!r} is missing a neutral row for some item.")
        neutral_values = np.asarray([neutral_map[item] for item in item_order], dtype=np.float64)
        for framing in framings:
            framing_map = _aligned(rows, framing, outcome)
            if set(framing_map) != set(item_order):
                raise ValueError(f"Outcome {outcome!r} is missing framing {framing!r} for an item.")
            values = np.asarray([framing_map[item] for item in item_order], dtype=np.float64)
            summary = bootstrap_mean_ci(values, rng, n_boot=n_boot)
            by_framing.append(
                {
                    "kind": kind,
                    "framing": framing,
                    "outcome": outcome,
                    "n_items": int(summary["n"]),
                    "mean": summary["mean"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                }
            )
            if framing == neutral:
                continue
            delta = paired_delta_ci(values, neutral_values, rng, n_boot=n_boot)
            record: dict[str, Any] = {
                "kind": kind,
                "framing": framing,
                "outcome": outcome,
                "n_items": int(delta["n"]),
                "mean_framing": delta["mean_treated"],
                "mean_neutral": delta["mean_control"],
                "mean_delta": delta["mean"],
                "ci_low": delta["ci_low"],
                "ci_high": delta["ci_high"],
                "mcnemar_b": "",
                "mcnemar_c": "",
                "mcnemar_p": "",
            }
            if outcome in binary_outcomes:
                exact = mcnemar_exact(values, neutral_values)
                record["mcnemar_b"] = exact["b"]
                record["mcnemar_c"] = exact["c"]
                record["mcnemar_p"] = exact["p"]
            paired.append(record)

    associations: list[dict[str, Any]] = []
    item_ids = np.asarray([str(row["item_id"]) for row in rows])
    framing_ids = np.asarray([str(row["framing"]) for row in rows])
    for outcome in outcomes:
        y = np.asarray([float(row[outcome]) for row in rows], dtype=np.float64)
        for predictor in predictors:
            x = np.asarray([float(row[predictor]) for row in rows], dtype=np.float64)
            for method in ("framing", "framing_and_item"):
                association = partial_association(
                    y,
                    x,
                    item_ids,
                    framing_ids,
                    rng,
                    n_boot=n_boot,
                    method=method,
                )
                association["kind"] = kind
                association["outcome"] = outcome
                association["predictor"] = predictor
                associations.append(association)
    return {"by_framing": by_framing, "paired": paired, "associations": associations}
