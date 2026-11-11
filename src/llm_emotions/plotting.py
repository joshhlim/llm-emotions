"""Figures for phase 01. Plot functions return figures and do not call ``show``."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from llm_emotions.data import Scenario, iter_scenario_controls
from llm_emotions.directions import DirectionFit, direction_matrix


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")


def plot_probe_accuracy(fit: DirectionFit) -> Figure:
    """Probe accuracy against layer. The dotted line marks the chosen layer."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("white")
    n_layers = len(fit.probe_accuracy[fit.emotions[0]])
    layer_ids = np.arange(n_layers)
    finite_values: list[np.ndarray] = []
    for emotion in fit.emotions:
        accuracy = fit.probe_accuracy[emotion]
        finite = accuracy[np.isfinite(accuracy)]
        if len(finite):
            finite_values.append(finite)
        ax.plot(layer_ids, accuracy, label=emotion, linewidth=2)
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1, label="chance")
    ax.axvline(fit.best_layer, color="0.3", linestyle=":", linewidth=1)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Cross-validated logistic probe accuracy")
    ax.set_title("Probe accuracy by layer on the keyword-heavy contrastive set")
    if finite_values:
        lowest = float(np.concatenate(finite_values).min())
        ax.set_ylim(max(0.0, min(0.35, lowest - 0.05)), 1.02)
    ax.legend(loc="lower right", ncol=2, frameon=False)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


@dataclass
class PcaPlot:
    figure: Figure
    explained_variance_ratio: np.ndarray
    coordinates: np.ndarray


def plot_direction_pca(fit: DirectionFit, *, seed: int) -> PcaPlot:
    """PCA of the mean-difference directions at the chosen layer.

    With eight points this is a picture of the fitted directions, not a test of
    a psychological circumplex. Read it as a map, and check it on another model
    before treating an axis as valence or arousal.
    """
    matrix = direction_matrix(fit, fit.best_layer)
    if matrix.shape[0] < 2:
        raise ValueError("PCA needs at least two emotion directions.")
    pca = PCA(n_components=2, random_state=seed)
    coordinates = pca.fit_transform(matrix)
    variance = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    fig.patch.set_facecolor("white")
    ax.axhline(0, color="0.85")
    ax.axvline(0, color="0.85")
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=60)
    for index, emotion in enumerate(fit.emotions):
        ax.annotate(
            emotion,
            coordinates[index],
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=11,
        )
    ax.set_xlabel(f"PC1 ({100 * variance[0]:.0f}% var)")
    ax.set_ylabel(f"PC2 ({100 * variance[1]:.0f}% var)")
    ax.set_title(f"Mean-difference directions at layer {fit.best_layer}")
    _clean_axes(ax)
    fig.tight_layout()
    return PcaPlot(figure=fig, explained_variance_ratio=variance, coordinates=coordinates)


def plot_scenario_scores(
    scores: dict[str, dict[str, float]],
    scenarios: list[Scenario],
    emotions: list[str],
    layer: int,
) -> Figure:
    """Grouped bars of scenario vs matched control at one layer."""
    paired = iter_scenario_controls(scenarios)
    n_pairs = len(paired)
    ncols = 3 if n_pairs > 1 else 1
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.25 * nrows),
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    flat = axes.ravel()
    x = np.arange(len(emotions))
    for ax, (treated, control) in zip(flat[:n_pairs], paired, strict=True):
        control_scores = [scores[control.id][emotion] for emotion in emotions]
        treated_scores = [scores[treated.id][emotion] for emotion in emotions]
        ax.bar(x - 0.18, control_scores, width=0.36, label="control", color="0.75")
        ax.bar(x + 0.18, treated_scores, width=0.36, label="scenario", color="#c44e52")
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45, ha="right", fontsize=8)
        ax.set_title(treated.pair.replace("_", " "))
        ax.axhline(0, color="0.4", linewidth=0.8)
        _clean_axes(ax)
    for ax in flat[n_pairs:]:
        ax.axis("off")
    flat[0].legend(frameon=False, loc="upper right", fontsize=8)
    flat[0].set_ylabel("projection")
    if nrows > 1:
        flat[ncols].set_ylabel("projection")
    fig.suptitle(f"Emotion-direction projections at layer {layer}", y=1.02)
    fig.tight_layout()
    return fig
