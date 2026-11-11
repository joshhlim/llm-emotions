"""Mean-difference directions and cross-validated logistic probes."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ActivationMap = dict[str, dict[str, np.ndarray]]


def set_seed(seed: int) -> None:
    """Pin Python, NumPy, and PyTorch. The probe and PCA also take this seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unit(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalise a vector. A zero vector stays zero."""
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, eps)


@dataclass
class DirectionFit:
    """Directions and probe scores indexed by hidden-state layer, including embeddings."""

    emotions: list[str]
    probe_accuracy: dict[str, np.ndarray]
    direction_accuracy: dict[str, np.ndarray]
    directions: dict[str, np.ndarray]
    best_layer: int
    n_pairs: dict[str, int]

    def mean_probe_at(self, layer: int) -> float:
        return float(np.mean([self.probe_accuracy[emotion][layer] for emotion in self.emotions]))

    def mean_direction_at(self, layer: int) -> float:
        return float(
            np.mean([self.direction_accuracy[emotion][layer] for emotion in self.emotions])
        )


def select_best_layer(probe_accuracy: dict[str, np.ndarray], emotions: list[str]) -> int:
    """Pick the layer with the highest mean probe accuracy.

    Layer 0 (embeddings) is ignored when any later layer was actually fit, so a
    lexical embedding match does not win by default. Ties go to the earlier layer.
    """
    stacked = np.vstack([probe_accuracy[emotion] for emotion in emotions])
    mean = np.empty(stacked.shape[1], dtype=np.float64)
    for index in range(stacked.shape[1]):
        column = stacked[:, index]
        finite = column[np.isfinite(column)]
        mean[index] = float(finite.mean()) if len(finite) else np.nan
    finite = np.isfinite(mean)
    later = finite.copy()
    if len(later):
        later[0] = False
    candidates = np.flatnonzero(later if later.any() else finite)
    if len(candidates) == 0:
        raise RuntimeError("No layer produced a finite probe accuracy.")
    best_offset = int(np.argmax(mean[candidates]))
    return int(candidates[best_offset])


def fit_directions_and_probes(
    activations: ActivationMap,
    *,
    seed: int,
    n_splits: int = 5,
    probe_C: float = 0.5,
    probe_max_iter: int = 2000,
    layers: list[int] | None = None,
) -> DirectionFit:
    """Fit a unit mean-difference direction and a logistic probe at each layer.

    The probe is a standardised logistic regression scored with stratified
    cross-validation, so the small contrastive set is not graded on its own
    training rows. Chance is 0.5.

    The direction score uses the same folds. The decision threshold is the
    midpoint of the two class means. A raw dot product with a residual is a poor
    threshold: the stream has a large positive mean, so the sign is almost
    always positive.

    Directions that are saved for later scoring are refit on every row. Layers
    that were not requested stay NaN (accuracy) or zero (direction).
    """
    emotions = list(activations)
    if not emotions:
        raise ValueError("No activations to fit.")

    sample = activations[emotions[0]]["emotion"]
    n_layers_plus = int(sample.shape[1])
    hidden_size = int(sample.shape[2])
    if layers is None:
        layer_ids = list(range(n_layers_plus))
    else:
        layer_ids = list(layers)
        for layer in layer_ids:
            if layer < 0 or layer >= n_layers_plus:
                raise ValueError(
                    f"Layer {layer} is outside 0..{n_layers_plus - 1} (0 is the embedding stream)."
                )

    probe_accuracy = {emotion: np.full(n_layers_plus, np.nan) for emotion in emotions}
    direction_accuracy = {emotion: np.full(n_layers_plus, np.nan) for emotion in emotions}
    directions = {
        emotion: np.zeros((n_layers_plus, hidden_size), dtype=np.float64) for emotion in emotions
    }
    n_pairs: dict[str, int] = {}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for emotion in emotions:
        positive = np.asarray(activations[emotion]["emotion"], dtype=np.float64)
        negative = np.asarray(activations[emotion]["neutral"], dtype=np.float64)
        if positive.shape != negative.shape:
            raise ValueError(f"{emotion}: emotion and neutral activations have different shapes.")
        if positive.shape[0] < n_splits:
            raise ValueError(
                f"{emotion}: need at least {n_splits} pairs for {n_splits}-fold CV, "
                f"found {positive.shape[0]}."
            )
        n_pairs[emotion] = int(positive.shape[0])
        labels = np.array([1] * len(positive) + [0] * len(negative))
        for layer in layer_ids:
            features = np.concatenate([positive[:, layer], negative[:, layer]], axis=0)
            fold_probe: list[float] = []
            fold_direction: list[float] = []
            for train_idx, test_idx in splitter.split(features, labels):
                x_train, x_test = features[train_idx], features[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=probe_max_iter,
                        C=probe_C,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                )
                classifier.fit(x_train, y_train)
                fold_probe.append(float(accuracy_score(y_test, classifier.predict(x_test))))

                mean_positive = x_train[y_train == 1].mean(axis=0)
                mean_negative = x_train[y_train == 0].mean(axis=0)
                direction = unit(mean_positive - mean_negative)
                midpoint = 0.5 * (mean_positive + mean_negative)
                projection = (x_test - midpoint) @ direction
                fold_direction.append(float(accuracy_score(y_test, (projection > 0).astype(int))))
            probe_accuracy[emotion][layer] = float(np.mean(fold_probe))
            direction_accuracy[emotion][layer] = float(np.mean(fold_direction))
            directions[emotion][layer] = unit(
                positive[:, layer].mean(axis=0) - negative[:, layer].mean(axis=0)
            )

    return DirectionFit(
        emotions=emotions,
        probe_accuracy=probe_accuracy,
        direction_accuracy=direction_accuracy,
        directions=directions,
        best_layer=select_best_layer(probe_accuracy, emotions),
        n_pairs=n_pairs,
    )


def direction_matrix(fit: DirectionFit, layer: int) -> np.ndarray:
    """Stack unit directions at ``layer`` into ``(n_emotions, hidden_size)``."""
    return np.stack([fit.directions[emotion][layer] for emotion in fit.emotions], axis=0)


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of row vectors. Rows are assumed to be unit length."""
    return matrix @ matrix.T
