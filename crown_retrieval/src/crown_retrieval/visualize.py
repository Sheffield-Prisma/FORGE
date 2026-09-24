"""Create class-similarity and t-SNE visualizations for labeled crown embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from .evaluate import load_labeled_index


def class_similarity(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate mean and standard deviation of pairwise cosine similarity per class pair."""
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    similarities = normalized @ normalized.T
    unique_labels = np.sort(np.unique(labels))
    means = np.zeros((len(unique_labels), len(unique_labels)))
    standard_deviations = np.zeros_like(means)
    for row, left_label in enumerate(unique_labels):
        left_indices = np.flatnonzero(labels == left_label)
        for column, right_label in enumerate(unique_labels):
            right_indices = np.flatnonzero(labels == right_label)
            values = similarities[np.ix_(left_indices, right_indices)]
            if row == column:
                values = values[~np.eye(len(left_indices), dtype=bool)]
            means[row, column] = values.mean() if values.size else 1.0
            standard_deviations[row, column] = values.std() if values.size else 0.0
    return means, standard_deviations, unique_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--annotation-id-column", default="id")
    parser.add_argument("--label-column", default="class_id")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.index / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    features, metadata = load_labeled_index(args.index, args.annotations, args.annotation_id_column, args.label_column, None)
    counts = metadata[args.label_column].value_counts()
    included_labels = counts[counts >= args.min_samples].index
    keep = metadata[args.label_column].isin(included_labels).to_numpy()
    features, labels = features[keep], metadata.loc[keep, args.label_column].to_numpy()
    if len(features) < 3:
        raise ValueError("Too few samples remain after the minimum-class-size filter.")

    means, standard_deviations, unique_labels = class_similarity(features, labels)
    annotation = np.array([[f"{mean:.2f}\n±{std:.2f}" for mean, std in zip(mean_row, std_row)] for mean_row, std_row in zip(means, standard_deviations)])
    figure, axis = plt.subplots(figsize=(max(8, len(unique_labels) * 0.6), max(7, len(unique_labels) * 0.55)))
    sns.heatmap(means, annot=annotation, fmt="", cmap="GnBu", vmin=0, vmax=1, xticklabels=unique_labels, yticklabels=unique_labels, ax=axis, cbar_kws={"label": "Mean pairwise cosine similarity"})
    axis.set_title("Class similarity: mean ± standard deviation")
    axis.set_xlabel(args.label_column)
    axis.set_ylabel(args.label_column)
    figure.tight_layout()
    figure.savefig(output / "class_similarity.png", dpi=180)
    plt.close(figure)

    valid_perplexity = min(args.perplexity, max(2, len(features) - 1))
    embedding_2d = TSNE(n_components=2, perplexity=valid_perplexity, init="pca", random_state=42, max_iter=1000).fit_transform(features)
    figure, axis = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        selection = labels == label
        axis.scatter(embedding_2d[selection, 0], embedding_2d[selection, 1], s=25, alpha=0.8, label=str(label))
    axis.set_title("t-SNE of masked-crown DINOv2 embeddings")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend(title=args.label_column, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output / "tsne.png", dpi=180)
    plt.close(figure)
    print(f"Saved visualizations to {output}")


if __name__ == "__main__":
    main()

