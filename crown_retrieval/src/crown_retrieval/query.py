"""Query a saved crown feature index and export ranked similar crowns."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np


def load_index(index_directory: Path) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    """Load embeddings and candidate geometries, verifying their row alignment."""
    embeddings = np.load(index_directory / "embeddings.npy")
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    candidates = gpd.read_file(index_directory / "candidates.gpkg").sort_values("retrieval_id").reset_index(drop=True)
    if len(embeddings) != len(candidates):
        raise ValueError("embeddings.npy and candidates.gpkg contain different row counts.")
    if not np.array_equal(candidates["retrieval_id"].to_numpy(), np.arange(len(candidates))):
        raise ValueError("candidates.gpkg does not have a contiguous retrieval_id column.")
    return embeddings, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query-id", required=True, help="Value from source_id by default, or retrieval_id with --id-column retrieval_id.")
    parser.add_argument("--id-column", choices=["source_id", "retrieval_id"], default="source_id")
    parser.add_argument("--top-k", type=int, default=100, help="Use 0 to export every non-query candidate.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    embeddings, candidates = load_index(args.index)
    values = candidates[args.id_column].astype(str)
    matches = np.flatnonzero(values == str(args.query_id))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {args.id_column}={args.query_id}, found {len(matches)}.")
    query_position = int(matches[0])
    similarities = embeddings @ embeddings[query_position]
    order = np.argsort(-similarities)
    order = order[order != query_position]
    if args.top_k:
        order = order[: args.top_k]
    results = candidates.iloc[order].copy().reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    results.insert(1, "similarity", similarities[order])
    output = args.output
    if output is None:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.query_id))
        output = args.index / f"retrieval_{safe_id}.gpkg"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_file(output, driver="GPKG")
    print(f"Saved {len(results)} ranked candidates to {output}")


if __name__ == "__main__":
    main()
