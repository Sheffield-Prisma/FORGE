"""Post-hoc open-set scoring methods and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score


@torch.inference_mode()
def extract_outputs(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract CLS features, logits, and labels in one deterministic pass."""
    model.eval()
    features, logits, labels = [], [], []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        batch_features = model.forward_features(images)
        features.append(batch_features.cpu().numpy())
        logits.append(model.head(batch_features).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(logits), np.concatenate(labels)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return numerically stable log-sum-exp energy scores."""
    scaled = logits / temperature
    maximum = scaled.max(axis=1, keepdims=True)
    return temperature * (maximum[:, 0] + np.log(np.exp(scaled - maximum).sum(axis=1)))


def negative_entropy(logits: np.ndarray) -> np.ndarray:
    probabilities = softmax(logits)
    return (probabilities * np.log(probabilities + 1e-12)).sum(axis=1)


@dataclass
class MahalanobisScorer:
    """Shared-covariance distance to the nearest known-class centroid."""

    means: np.ndarray
    precision: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, regularization: float) -> "MahalanobisScorer":
        classes = np.unique(labels)
        means = np.stack([features[labels == label].mean(axis=0) for label in classes])
        centered = np.concatenate(
            [features[labels == label] - means[index] for index, label in enumerate(classes)]
        )
        covariance = np.cov(centered, rowvar=False)
        precision = np.linalg.pinv(covariance + regularization * np.eye(covariance.shape[0]))
        return cls(means=means, precision=precision)

    def score(self, features: np.ndarray) -> np.ndarray:
        deltas = features[:, None, :] - self.means[None, :, :]
        squared_distance = np.einsum("ncd,de,nce->nc", deltas, self.precision, deltas)
        return -squared_distance.min(axis=1)

    def nearest_class_index(self, features: np.ndarray) -> np.ndarray:
        deltas = features[:, None, :] - self.means[None, :, :]
        squared_distance = np.einsum("ncd,de,nce->nc", deltas, self.precision, deltas)
        return squared_distance.argmin(axis=1)


@dataclass
class KNNScorer:
    """Mean Euclidean distance to k normalized known training features."""

    normalized_features: np.ndarray
    k: int

    @classmethod
    def fit(cls, features: np.ndarray, k: int) -> "KNNScorer":
        if not 1 <= k <= len(features):
            raise ValueError(f"k must be between 1 and {len(features)}, received {k}.")
        normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        return cls(normalized_features=normalized, k=k)

    def score(self, features: np.ndarray) -> np.ndarray:
        normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        scores = []
        for feature in normalized:
            distances = np.linalg.norm(self.normalized_features - feature, axis=1)
            nearest = np.partition(distances, self.k - 1)[: self.k]
            scores.append(-nearest.mean())
        return np.asarray(scores)


@dataclass
class ViMScorer:
    """Residual-feature virtual-logit matching, calibrated on training data."""

    mean: np.ndarray
    residual_basis: np.ndarray
    alpha: float

    @classmethod
    def fit(
        cls, features: np.ndarray, train_logits: np.ndarray, principal_dimensions: int
    ) -> "ViMScorer":
        if not 1 <= principal_dimensions < features.shape[1]:
            raise ValueError("principal_dimensions must be in [1, feature_dimension).")
        mean = features.mean(axis=0)
        centered = features - mean
        covariance = centered.T @ centered / len(centered)
        _, eigenvectors = np.linalg.eigh(covariance)  # ascending eigenvalues
        # The complement of the high-variance principal subspace is the residual space.
        residual_basis = eigenvectors[:, : features.shape[1] - principal_dimensions]
        residual_norm = np.linalg.norm(centered @ residual_basis, axis=1)
        alpha = train_logits.max(axis=1).mean() / max(residual_norm.mean(), 1e-12)
        return cls(mean=mean, residual_basis=residual_basis, alpha=float(alpha))

    def score(self, features: np.ndarray, logits: np.ndarray) -> np.ndarray:
        residual_norm = np.linalg.norm((features - self.mean) @ self.residual_basis, axis=1)
        return energy(logits) - self.alpha * residual_norm


def react_energy(
    features: np.ndarray, head: Callable[[torch.Tensor], torch.Tensor], device: torch.device, clipping: np.ndarray
) -> np.ndarray:
    """Clip each feature dimension at its train-set 90th percentile, then score energy."""
    clipped = np.minimum(features, clipping)
    with torch.inference_mode():
        logits = head(torch.from_numpy(clipped).float().to(device)).cpu().numpy()
    return energy(logits)


def osr_auc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(labels, scores))


def oscr(known_scores: np.ndarray, known_is_correct: np.ndarray, unknown_scores: np.ndarray) -> float:
    """Compute Open-Set Classification Rate without arbitrary threshold gridding."""
    thresholds = np.r_[np.inf, np.unique(np.r_[known_scores, unknown_scores])[::-1], -np.inf]
    correct_classification_rate = []
    false_positive_rate = []
    for threshold in thresholds:
        correct_classification_rate.append(
            np.mean((known_scores >= threshold) & known_is_correct)
        )
        false_positive_rate.append(np.mean(unknown_scores >= threshold))
    return float(auc(false_positive_rate, correct_classification_rate))


def score_all_methods(
    train_features: np.ndarray,
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    known_features: np.ndarray,
    known_logits: np.ndarray,
    unknown_features: np.ndarray,
    unknown_logits: np.ndarray,
    *,
    model_head: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    knn_k: int,
    covariance_regularization: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit all train-dependent scorers and return known/unknown scores."""
    maha = MahalanobisScorer.fit(train_features, train_labels, covariance_regularization)
    knn = KNNScorer.fit(train_features, knn_k)
    vim = ViMScorer.fit(train_features, train_logits, principal_dimensions=len(np.unique(train_labels)))
    clipping = np.percentile(train_features, 90, axis=0)
    return {
        "MSP": (softmax(known_logits).max(axis=1), softmax(unknown_logits).max(axis=1)),
        "MaxLogit": (known_logits.max(axis=1), unknown_logits.max(axis=1)),
        "Energy": (energy(known_logits), energy(unknown_logits)),
        "Entropy": (negative_entropy(known_logits), negative_entropy(unknown_logits)),
        "Mahalanobis": (maha.score(known_features), maha.score(unknown_features)),
        "KNN": (knn.score(known_features), knn.score(unknown_features)),
        "ViM": (vim.score(known_features, known_logits), vim.score(unknown_features, unknown_logits)),
        "ReAct+Energy": (
            react_energy(known_features, model_head, device, clipping),
            react_energy(unknown_features, model_head, device, clipping),
        ),
    }

