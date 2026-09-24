"""Measure crown-level precision, recall, and F1 from georeferenced held-out predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box


def load_target_crowns(path: Path, species_field: str, species_substring: str, crs) -> gpd.GeoDataFrame:
    """Load valid target crowns and project them to the prediction CRS."""
    crowns = gpd.read_file(path)
    crowns = crowns[crowns[species_field].astype(str).str.contains(species_substring, case=False, na=False)]
    crowns = crowns[crowns.geometry.notna() & ~crowns.geometry.is_empty].copy()
    crowns["geometry"] = crowns.geometry.buffer(0)
    return crowns[crowns.geometry.is_valid & ~crowns.geometry.is_empty].to_crs(crs).reset_index(drop=True)


def clip_to_raster_footprint(crowns: gpd.GeoDataFrame, raster_path: Path) -> gpd.GeoDataFrame:
    """Avoid counting crowns from other plots as false negatives."""
    with rasterio.open(raster_path) as raster:
        bounds, raster_crs = raster.bounds, raster.crs
    footprint = gpd.GeoSeries([box(bounds.left, bounds.bottom, bounds.right, bounds.top)], crs=raster_crs).to_crs(crowns.crs).iloc[0]
    return crowns[crowns.geometry.centroid.within(footprint)].reset_index(drop=True)


def polygon_iou(left, right) -> float:
    intersection = left.intersection(right).area
    if intersection <= 0:
        return 0.0
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def match_predictions(predictions: gpd.GeoDataFrame, crowns: gpd.GeoDataFrame, mode: str, iou_threshold: float) -> tuple[int, int, int, list[bool], list[bool]]:
    """Greedily match one prediction to at most one target crown."""
    prediction_matches = [False] * len(predictions)
    crown_matches = [False] * len(crowns)
    if crowns.empty:
        return 0, len(predictions), 0, prediction_matches, crown_matches
    spatial_index = crowns.sindex
    order = predictions.index.tolist()
    if "confidence" in predictions.columns:
        order = predictions["confidence"].fillna(0).sort_values(ascending=False).index.tolist()
    for prediction_index in order:
        prediction = predictions.geometry.loc[prediction_index]
        best_index, best_score = -1, 0.0
        for crown_index in spatial_index.intersection(prediction.bounds):
            if crown_matches[crown_index]:
                continue
            crown = crowns.geometry.iloc[crown_index]
            score = (
                1.0 if prediction.centroid.within(crown) or crown.centroid.within(prediction)
                else 0.0
            ) if mode == "centroid" else polygon_iou(prediction, crown)
            if score > best_score:
                best_index, best_score = crown_index, score
        threshold = 1.0 if mode == "centroid" else iou_threshold
        if best_index >= 0 and best_score >= threshold:
            prediction_matches[prediction_index] = True
            crown_matches[best_index] = True
    true_positive = sum(crown_matches)
    false_positive = len(predictions) - sum(prediction_matches)
    false_negative = len(crowns) - true_positive
    return true_positive, false_positive, false_negative, prediction_matches, crown_matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--rasters", type=Path, nargs="+", required=True, help="Held-out rasters, in the same order as predictions.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--species-field", default="NOMBRE_CIE")
    parser.add_argument("--species-substring", default="Dipteryx")
    parser.add_argument("--mode", choices=["iou", "centroid"], default="iou")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("crown_metrics.json"))
    args = parser.parse_args()
    if len(args.predictions) != len(args.rasters):
        parser.error("--predictions and --rasters must have the same number of paths.")

    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    for prediction_path, raster_path in zip(args.predictions, args.rasters):
        predictions = gpd.read_file(prediction_path)
        predictions = predictions[predictions.geometry.notna() & ~predictions.geometry.is_empty].copy()
        if "confidence" in predictions.columns and args.minimum_confidence:
            predictions = predictions[predictions["confidence"].fillna(0) >= args.minimum_confidence]
        predictions["geometry"] = predictions.geometry.buffer(0)
        predictions = predictions[predictions.geometry.is_valid & ~predictions.geometry.is_empty].reset_index(drop=True)
        crowns = clip_to_raster_footprint(
            load_target_crowns(args.ground_truth, args.species_field, args.species_substring, predictions.crs), raster_path
        )
        true_positive, false_positive, false_negative, _, _ = match_predictions(predictions, crowns, args.mode, args.iou_threshold)
        totals["true_positive"] += true_positive
        totals["false_positive"] += false_positive
        totals["false_negative"] += false_negative
        print(f"{prediction_path.name}: TP={true_positive} FP={false_positive} FN={false_negative}")

    precision = totals["true_positive"] / max(totals["true_positive"] + totals["false_positive"], 1)
    recall = totals["true_positive"] / max(totals["true_positive"] + totals["false_negative"], 1)
    summary = {
        **totals,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 4),
        "match_mode": args.mode,
        "iou_threshold": args.iou_threshold if args.mode == "iou" else None,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

