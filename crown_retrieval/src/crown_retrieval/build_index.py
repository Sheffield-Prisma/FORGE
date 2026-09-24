"""Build a reusable DINOv2 feature index from a raster and YOLO crown candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from .embeddings import embed_crop, load_dinov2, polygon_crop, preprocessing, resolve_device


def resolve_identifier_column(frame: gpd.GeoDataFrame, requested: str | None) -> str | None:
    """Resolve an explicit identifier column or a conventional FID field, if present."""
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Requested identifier column '{requested}' is absent.")
        return requested
    return next((column for column in frame.columns if column.lower() == "fid"), None)


def validated_candidates(path: Path, raster_crs, identifier_column: str | None) -> gpd.GeoDataFrame:
    """Load candidates, preserve a stable source ID, and align their CRS to the raster."""
    candidates = gpd.read_file(path)
    candidates = candidates[candidates.geometry.notna() & ~candidates.geometry.is_empty].copy()
    candidates["geometry"] = candidates.geometry.buffer(0)
    candidates = candidates[candidates.geometry.is_valid & ~candidates.geometry.is_empty].copy()
    if candidates.empty:
        raise ValueError("The candidate layer has no valid geometries.")
    if candidates.crs is None:
        raise ValueError("The candidate layer has no CRS.")
    if raster_crs is None:
        raise ValueError("The raster has no CRS.")
    source_ids = candidates[identifier_column].astype(str) if identifier_column else candidates.index.astype(str)
    candidates = candidates.to_crs(raster_crs).reset_index(drop=True)
    # Reserve these names for the stable index schema even if the source layer used them.
    candidates = candidates.drop(columns=[column for column in ("source_id", "retrieval_id") if column in candidates.columns])
    candidates.insert(0, "source_id", source_ids.to_numpy())
    return candidates


def build_index(args: argparse.Namespace) -> Path:
    """Extract one masked DINOv2 embedding per valid predicted crown."""
    raster_path, candidate_path, output_directory = Path(args.raster), Path(args.candidates), Path(args.output)
    if output_directory.exists() and any(output_directory.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_directory} is not empty. Pass --overwrite to replace index files.")
    output_directory.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        # Only files owned by this index format are replaced; source imagery is never touched.
        for name in ("embeddings.npy", "candidates.gpkg", "failures.csv", "index_metadata.json"):
            destination = output_directory / name
            if destination.exists():
                destination.unlink()
    device = resolve_device(args.device)
    with rasterio.open(raster_path) as raster:
        candidates_raw = gpd.read_file(candidate_path)
        identifier_column = resolve_identifier_column(candidates_raw, args.id_column)
        # Reload in the helper so all geometry validity checks are applied consistently.
        candidates = validated_candidates(candidate_path, raster.crs, identifier_column)
        model = load_dinov2(args.model, device)
        transform = preprocessing(args.image_size)
        features, kept_positions, failures = [], [], []
        for position, row in candidates.iterrows():
            image = polygon_crop(row.geometry, raster, args.background_value)
            if image is None:
                failures.append({"source_id": row.source_id, "reason": "crop_failed"})
                continue
            try:
                features.append(embed_crop(image, model, transform, device))
                kept_positions.append(position)
            except Exception as error:
                failures.append({"source_id": row.source_id, "reason": str(error)})
        if not features:
            raise RuntimeError("No candidate embeddings were extracted.")

    indexed = candidates.iloc[kept_positions].copy().reset_index(drop=True)
    indexed.insert(0, "retrieval_id", np.arange(len(indexed), dtype=np.int64))
    embeddings = np.stack(features).astype(np.float32)
    np.save(output_directory / "embeddings.npy", embeddings)
    indexed.to_file(output_directory / "candidates.gpkg", driver="GPKG")
    pd.DataFrame(failures, columns=["source_id", "reason"]).to_csv(output_directory / "failures.csv", index=False)
    metadata = {
        "raster": str(raster_path.resolve()),
        "candidate_layer": str(candidate_path.resolve()),
        "model": args.model,
        "image_size": args.image_size,
        "background_value": args.background_value,
        "identifier_column": identifier_column,
        "indexed_candidates": len(indexed),
        "failed_candidates": len(failures),
        "embedding_dimension": int(embeddings.shape[1]),
    }
    (output_directory / "index_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Indexed {len(indexed)} crowns with {len(failures)} failures at {output_directory}")
    return output_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", required=True, help="Original orthomosaic TIFF containing the candidate crowns.")
    parser.add_argument("--candidates", required=True, help="YOLO crown predictions in GPKG, GeoJSON, or Shapefile form.")
    parser.add_argument("--output", required=True, help="Directory that will receive embeddings.npy and candidates.gpkg.")
    parser.add_argument("--id-column", default=None, help="Candidate ID column; defaults to case-insensitive FID, then source row index.")
    parser.add_argument("--model", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--background-value", type=int, default=124)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    build_index(parser.parse_args())


if __name__ == "__main__":
    main()
