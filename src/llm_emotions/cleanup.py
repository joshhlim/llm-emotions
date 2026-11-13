"""Remove the component shared by every emotion-versus-neutral contrast.

Phase 01 directions sit at cosine 0.5 to 0.85 with each other, including
anger with joy. That pattern is what a shared emotional-versus-neutral axis
looks like. Cleanup does two things, in order:

1. Subtract the mean difference across emotions, so a component present in
   every contrast is taken out once.
2. Project the remaining vectors off the unit vector in that same mean
   direction. Emotions that carried more than the average share lose the rest.

Vectors are then scaled back to unit length. A vector that was only the shared
component becomes zero and stays zero. Cosine similarity is reported before
and after. Cleanup does not force the eight directions to be orthogonal:
emotions can still cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit_last(matrix: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """L2-normalise along the last axis. Near-zero rows stay zero.

    :func:`llm_emotions.directions.unit` divides by ``eps`` and can inflate a
    tiny vector. Cleanup uses a hard threshold instead, because a contrast
    that was entirely the shared component should not be rescaled into noise.
    """
    values = np.asarray(matrix, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > tol)


@dataclass(frozen=True)
class CleanedDifferences:
    """Unit directions before cleanup, after mean subtraction, and after both steps.

    ``deltas`` input shape is ``(n_emotions, hidden)`` or
    ``(n_emotions, n_layers, hidden)``. Every returned array keeps that shape,
    except ``affect``, which has no emotion axis.
    """

    raw: np.ndarray
    mean_centered: np.ndarray
    cleaned: np.ndarray
    affect: np.ndarray


def clean_mean_differences(deltas: np.ndarray) -> CleanedDifferences:
    """Subtract the cross-emotion mean and project that direction out.

    ``deltas`` holds unnormalised mean differences, one row per emotion. The
    affect direction is their mean: equal weight per emotion, not per token.
    """
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim not in (2, 3):
        raise ValueError(
            "deltas must have shape (n_emotions, hidden) or (n_emotions, n_layers, hidden)."
        )
    if values.shape[0] < 2:
        raise ValueError("Cleanup needs at least two emotions so the shared mean is defined.")
    affect = values.mean(axis=0)
    affect_u = unit_last(affect)
    centered = values - affect
    # Broadcast the affect direction over the emotion axis.
    projected = np.sum(centered * affect_u, axis=-1, keepdims=True) * affect_u
    orthogonal = centered - projected
    return CleanedDifferences(
        raw=unit_last(values),
        mean_centered=unit_last(centered),
        cleaned=unit_last(orthogonal),
        affect=affect_u,
    )


def cosine_rows(matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of vectors stored along the last axis.

    ``matrix`` is ``(n, hidden)`` or ``(n, n_layers, hidden)``. The result is
    ``(n, n)`` or ``(n_layers, n, n)``.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim == 2:
        return values @ values.T
    if values.ndim == 3:
        return np.einsum("elh,flh->lef", values, values)
    raise ValueError("matrix must have shape (n, hidden) or (n, n_layers, hidden).")


def off_diagonal_summary(cosine: np.ndarray) -> dict[str, float]:
    """Mean, min, and max of the off-diagonal entries of one cosine matrix."""
    if cosine.ndim != 2 or cosine.shape[0] != cosine.shape[1]:
        raise ValueError("cosine must be a square matrix.")
    mask = ~np.eye(cosine.shape[0], dtype=bool)
    values = cosine[mask]
    if values.size == 0:
        empty = float("nan")
        return {
            "mean_off_diagonal": empty,
            "min_off_diagonal": empty,
            "max_off_diagonal": empty,
        }
    return {
        "mean_off_diagonal": float(values.mean()),
        "min_off_diagonal": float(values.min()),
        "max_off_diagonal": float(values.max()),
    }


def pairwise_accuracy_layers(
    directions: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score one direction per layer by whether the emotional sentence ranks higher.

    ``directions`` is ``(n_layers, hidden)``. ``positive`` and ``negative`` are
    ``(n_pairs, n_layers, hidden)`` and correspond row for row. Accuracy is the
    fraction of pairs whose emotional residual projects further along the
    direction than the matched neutral. A zero direction scores NaN: there is
    nothing to rank with. Ties (equal projections) count as misses.
    """
    dirs = np.asarray(directions, dtype=np.float64)
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.shape != neg.shape:
        raise ValueError("positive and negative activations must share a shape.")
    if dirs.shape != pos.shape[1:]:
        raise ValueError(
            f"directions shape {dirs.shape} does not match activations {pos.shape[1:]}."
        )
    margins = np.einsum("nlh,lh->nl", pos - neg, dirs)
    accuracy = np.mean(margins > 0.0, axis=0)
    norms = np.linalg.norm(dirs, axis=-1)
    accuracy = np.where(norms > 1e-8, accuracy, np.nan)
    return accuracy.astype(np.float64), margins.mean(axis=0).astype(np.float64)


def select_layer(scores: np.ndarray, *, eligible: np.ndarray | None = None) -> int:
    """Highest score. Layer 0 is skipped when any later eligible layer is finite.

    Ties go to the earlier layer. ``eligible`` is a boolean mask over layers;
    ``None`` means every layer.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one value per layer.")
    mask = np.isfinite(values)
    if eligible is not None:
        allowed = np.asarray(eligible, dtype=bool)
        if allowed.shape != values.shape:
            raise ValueError("eligible must be a boolean mask with one entry per layer.")
        mask &= allowed
    later = mask.copy()
    if later.size:
        later[0] = False
    candidates = np.flatnonzero(later if later.any() else mask)
    if candidates.size == 0:
        raise RuntimeError("No layer produced a finite score.")
    best_offset = int(np.argmax(values[candidates]))
    return int(candidates[best_offset])
