"""Evaluate labeled crown embeddings with mean average precision and Top-K retrieval metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .query import load_index


def load_labeled_index(
    index_directory: Path, annotations_path: Path, annotation_id_column: str, label_column: str, month_column: str | None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Align embeddings with external labels through the stable source_id column."""
    embeddings, candidates = load_index(index_directory)
    annotations = pd.read_csv(annotations_path)
    required = {annotation_id_column, label_column}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Annotation columns are missing: {', '.join(sorted(missing))}")
    annotations = annotations.copy()
    annotations["source_id"] = annotations[annotation_id_column].astype(str)
    if annotations["source_id"].duplicated().any():
        raise ValueError("Annotation IDs must be unique for retrieval evaluation.")
    columns = ["source_id", label_column] + ([month_column] if month_column else [])
    metadata = candidates[["retrieval_id", "source_id"]].merge(annotations[columns], on="source_id", how="inner", validate="one_to_one")
    if len(metadata) < 2:
        raise ValueError("Fewer than two indexed candidates have labels after joining annotations.")
    metadata = metadata.sort_values("retrieval_id").reset_index(drop=True)
    return embeddings[metadata["retrieval_id"].to_numpy()], metadata


def retrieval_metrics(embeddings: np.ndarray, labels: np.ndarray, maximum_k: int) -> tuple[pd.DataFrame, int]:
    """Calculate AP, precision, and recall at K for every query with a relevant neighbor."""
    features = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    similarities = features @ features.T
    np.fill_diagonal(similarities, -np.inf)
    max_k = min(maximum_k, len(features) - 1)
    order = np.argsort(-similarities, axis=1)[:, : len(features) - 1]
    records = []
    for query_index, ranked_indices in enumerate(order):
        relevant_total = int(np.sum(labels == labels[query_index]) - 1)
        if relevant_total < 1:
            continue
        hits = (labels[ranked_indices] == labels[query_index]).astype(int)
        cumulative_hits = np.cumsum(hits)
        ranks = np.arange(1, len(hits) + 1)
        precision = cumulative_hits / ranks
        recall = cumulative_hits / relevant_total
        average_precision = float((precision * hits).sum() / relevant_total)
        records.append(
            {
                "query_row": query_index,
                "label": labels[query_index],
                "average_precision": average_precision,
                "precision_at_k": float(cumulative_hits[max_k - 1] / max_k),
                "recall_at_k": float(cumulative_hits[max_k - 1] / relevant_total),
                "precision_curve": precision[:max_k],
                "recall_curve": recall[:max_k],
            }
        )
    if not records:
        raise ValueError("No label has at least two indexed candidates.")
    return pd.DataFrame(records), max_k


def class_summary(query_metrics: pd.DataFrame, maximum_k: int) -> pd.DataFrame:
    """Summarize mAP and the class-specific F1-optimal retrieval depth."""
    summaries = []
    for label, group in query_metrics.groupby("label", sort=True):
        mean_precision = group["precision_at_k"].mean()
        mean_recall = group["recall_at_k"].mean()
        f1 = 2 * mean_precision * mean_recall / max(mean_precision + mean_recall, 1e-12)
        precision_curve = np.vstack(group["precision_curve"].to_numpy()).mean(axis=0)
        recall_curve = np.vstack(group["recall_curve"].to_numpy()).mean(axis=0)
        f1_curve = 2 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 1e-12)
        best_index = int(f1_curve.argmax())
        summaries.append(
            {
                "label": label,
                "query_count": int(len(group)),
                "mean_average_precision": float(group["average_precision"].mean()),
                "best_k": best_index + 1,
                "best_f1": float(f1_curve[best_index]),
                f"precision_at_{maximum_k}": float(mean_precision),
                f"recall_at_{maximum_k}": float(mean_recall),
                f"f1_at_{maximum_k}": float(f1),
            }
        )
    return pd.DataFrame(summaries).sort_values("mean_average_precision", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--annotation-id-column", default="id")
    parser.add_argument("--label-column", default="class_id")
    parser.add_argument("--month-column", default=None, help="Optional annotation month column for monthly mAP output.")
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.index / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    embeddings, metadata = load_labeled_index(
        args.index, args.annotations, args.annotation_id_column, args.label_column, args.month_column
    )
    queries, actual_k = retrieval_metrics(embeddings, metadata[args.label_column].to_numpy(), args.max_k)
    queries.insert(0, "source_id", metadata.iloc[queries["query_row"].to_numpy()]["source_id"].to_numpy())
    if args.month_column:
        queries.insert(2, args.month_column, metadata.iloc[queries["query_row"].to_numpy()][args.month_column].to_numpy())
    classes = class_summary(queries, actual_k)
    queries.drop(columns=["precision_curve", "recall_curve"]).to_csv(output / "query_metrics.csv", index=False)
    classes.to_csv(output / "class_metrics.csv", index=False)
    summary = {
        "labeled_candidates": int(len(metadata)),
        "queries_with_relevant_neighbor": int(len(queries)),
        "maximum_k": actual_k,
        "mean_average_precision": float(queries["average_precision"].mean()),
        "macro_class_map": float(classes["mean_average_precision"].mean()),
    }
    if args.month_column:
        monthly = queries.dropna(subset=[args.month_column]).groupby(args.month_column)["average_precision"].agg(["count", "mean"])
        monthly.rename(columns={"mean": "mean_average_precision"}).to_csv(output / "monthly_metrics.csv")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(max(8, len(classes) * 0.35), 5))
    axis.bar(classes["label"].astype(str), classes["mean_average_precision"], color="#2E8B57")
    axis.set_ylim(0, 1)
    axis.set_xlabel(args.label_column)
    axis.set_ylabel("Mean average precision")
    axis.set_title("Per-class crown retrieval performance")
    axis.tick_params(axis="x", rotation=70)
    figure.tight_layout()
    figure.savefig(output / "class_map.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
