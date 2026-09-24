"""Build scale-normalized YOLO-seg tiles for single-class Shihuahuaco detection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from .config import DatasetConfig, load_dataset_config


def iter_polygons(geometry):
    """Yield Polygon members from a Polygon, MultiPolygon, or GeometryCollection."""
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for member in geometry.geoms:
            yield from iter_polygons(member)


def load_crowns(shapefile: Path, species_field: str, species_substring: str, target: bool) -> gpd.GeoDataFrame:
    """Load, filter, and repair crown annotations without assuming their CRS."""
    crowns = gpd.read_file(shapefile)
    if species_field not in crowns.columns:
        raise ValueError(f"'{species_field}' was not found in {shapefile}.")
    matches_target = crowns[species_field].astype(str).str.contains(species_substring, case=False, na=False)
    crowns = crowns[matches_target if target else ~matches_target].copy()
    crowns = crowns[crowns.geometry.notna() & ~crowns.geometry.is_empty].copy()
    invalid = ~crowns.geometry.is_valid
    if invalid.any():
        crowns.loc[invalid, "geometry"] = crowns.loc[invalid, "geometry"].buffer(0)
    return crowns[crowns.geometry.is_valid & ~crowns.geometry.is_empty].copy()


def find_tiffs(directory: Path) -> list[Path]:
    """Find TIFFs recursively and return their paths in deterministic order."""
    files = {path for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF") for path in directory.rglob(pattern)}
    return sorted(files)


def site_name(tif_path: Path, year: int) -> str:
    """Derive the study's site identifier from the source file naming convention."""
    if year == 2025:
        parts = tif_path.stem.split("_")
        return f"2025-{parts[1] if len(parts) > 1 else 'unknown'}"
    return tif_path.stem.split("_")[0]


def window_offsets(length: int, tile_pixels: int, stride_pixels: int) -> list[int]:
    """Cover the full raster dimension, including the final edge-aligned window."""
    if length <= tile_pixels:
        return [0]
    offsets = list(range(0, length - tile_pixels + 1, stride_pixels))
    final_offset = length - tile_pixels
    if offsets[-1] != final_offset:
        offsets.append(final_offset)
    return offsets


def yolo_lines(
    target_crowns: list[tuple], window: Window, transform, minimum_fraction: float
) -> list[str]:
    """Clip target polygons to a tile and encode sufficiently complete masks in YOLO format."""
    world_window = box(*rasterio.windows.bounds(window, transform))
    inverse_transform = ~transform
    lines = []
    for geometry, full_area in target_crowns:
        if not geometry.intersects(world_window):
            continue
        clipped = geometry.intersection(world_window)
        if clipped.is_empty or clipped.area / full_area < minimum_fraction:
            continue
        for polygon in iter_polygons(clipped):
            x_coordinates, y_coordinates = polygon.exterior.coords.xy
            normalized = []
            for x, y in zip(x_coordinates, y_coordinates):
                column, row = inverse_transform * (x, y)
                normalized.append(
                    (
                        min(max((column - window.col_off) / window.width, 0.0), 1.0),
                        min(max((row - window.row_off) / window.height, 0.0), 1.0),
                    )
                )
            if len(normalized) >= 3:
                coordinates = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
                lines.append(f"0 {coordinates}")
    return lines


def max_visible_fraction(target_crowns: list[tuple], window: Window, transform) -> float:
    """Return the largest fraction of any target crown visible inside a tile."""
    world_window = box(*rasterio.windows.bounds(window, transform))
    fractions = [
        geometry.intersection(world_window).area / full_area
        for geometry, full_area in target_crowns
        if full_area > 0 and geometry.intersects(world_window)
    ]
    return max(fractions, default=0.0)


def count_other_crowns(other_crowns: list, window: Window, transform) -> int:
    """Use annotated non-target crowns to prioritize hard negative candidates."""
    world_window = box(*rasterio.windows.bounds(window, transform))
    return sum(geometry.intersects(world_window) for geometry in other_crowns)


def read_rgb(source, window: Window) -> np.ndarray:
    """Read three bands and safely normalize non-uint8 input for JPEG output."""
    array = source.read([1, 2, 3], window=window)
    if array.dtype != np.uint8:
        values = array.astype(np.float32)
        lower, upper = np.percentile(values, (2, 98))
        values = np.clip((values - lower) / max(upper - lower, 1e-6) * 255, 0, 255)
        array = values.astype(np.uint8)
    return np.transpose(array, (1, 2, 0))


def save_tile(rgb: np.ndarray, image_path: Path, config: DatasetConfig) -> None:
    """Resample every ground-size tile to a fixed pixel scale and save as JPEG."""
    resized = cv2.resize(rgb, (config.output_pixels, config.output_pixels), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(image_path), resized[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])


def _state_file(state_directory: Path, tif_path: Path) -> Path:
    digest = hashlib.md5(str(tif_path).encode("utf-8")).hexdigest()[:16]
    return state_directory / f"{tif_path.stem[:60]}__{digest}.json"


def save_state(state_directory: Path, tif_path: Path, positives: list[dict], negatives: list[dict]) -> None:
    """Write per-TIF state atomically so interrupted dataset builds can resume."""
    destination = _state_file(state_directory, tif_path)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps({"positives": positives, "negatives": negatives}), encoding="utf-8")
    temporary.replace(destination)


def load_state(state_directory: Path, tif_path: Path) -> tuple[list[dict], list[dict]] | None:
    """Load a completed TIF state; corrupt state is treated as unfinished work."""
    path = _state_file(state_directory, tif_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["positives"], payload["negatives"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _record(key: str, site: str, year: int, is_positive: bool, source_tif: Path, **extra) -> dict:
    return {
        "key": key,
        "site": site,
        "year": year,
        "is_positive": int(is_positive),
        "source_tif": str(source_tif),
        **extra,
    }


def process_tif(
    tif_path: Path,
    year: int,
    target_annotations: gpd.GeoDataFrame,
    other_annotations: gpd.GeoDataFrame | None,
    image_directory: Path,
    label_directory: Path,
    config: DatasetConfig,
) -> tuple[list[dict], list[dict]]:
    """Emit positive tiles and collect negative candidates for one source raster."""
    positives, negative_candidates = [], []
    try:
        source = rasterio.open(tif_path)
    except rasterio.errors.RasterioError as error:
        print(f"[WARNING] Skipping unreadable raster {tif_path.name}: {error}")
        return positives, negative_candidates

    with source:
        if source.crs is None or source.count < 3:
            print(f"[WARNING] Skipping {tif_path.name}: missing CRS or RGB bands.")
            return positives, negative_candidates
        ground_sampling_distance = abs(source.transform.a)
        if not 0.005 <= ground_sampling_distance <= 0.5:
            print(f"[WARNING] Skipping {tif_path.name}: implausible GSD {ground_sampling_distance}.")
            return positives, negative_candidates
        try:
            footprint = box(*source.bounds)
            targets = target_annotations.to_crs(source.crs)
            targets = targets[targets.intersects(footprint)]
            target_crowns = [(geometry, geometry.area) for geometry in targets.geometry]
            if other_annotations is not None:
                others = other_annotations.to_crs(source.crs)
                other_crowns = list(others[others.intersects(footprint)].geometry)
            else:
                other_crowns = []
        except Exception as error:
            print(f"[WARNING] Skipping {tif_path.name}: per-raster CRS conversion failed: {error}")
            return positives, negative_candidates

        tile_pixels = int(round(config.tile_size_m / ground_sampling_distance))
        stride_pixels = max(1, int(round(tile_pixels * (1 - config.overlap))))
        site, stem = site_name(tif_path, year), tif_path.stem
        for row_offset in window_offsets(source.height, tile_pixels, stride_pixels):
            for column_offset in window_offsets(source.width, tile_pixels, stride_pixels):
                width = min(tile_pixels, source.width - column_offset)
                height = min(tile_pixels, source.height - row_offset)
                window = Window(column_offset, row_offset, width, height)
                try:
                    if (source.dataset_mask(window=window) == 0).mean() > config.maximum_nodata_fraction:
                        continue
                    key = f"{year}__{site}__{stem}__r{row_offset}_c{column_offset}"
                    lines = yolo_lines(target_crowns, window, source.transform, config.minimum_crown_fraction)
                    if lines:
                        save_tile(read_rgb(source, window), image_directory / f"{key}.jpg", config)
                        (label_directory / f"{key}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                        positives.append(_record(key, site, year, True, tif_path, other_crown_count=0))
                    elif max_visible_fraction(target_crowns, window, source.transform) <= config.negative_exclusion_fraction:
                        negative_candidates.append(
                            _record(
                                key, site, year, False, tif_path, column_offset=column_offset,
                                row_offset=row_offset, width=width, height=height,
                                other_crown_count=count_other_crowns(other_crowns, window, source.transform),
                            )
                        )
                except Exception as error:
                    print(f"[WARNING] Tile read failed in {tif_path.name}: {error}")

        if config.include_centered_tiles:
            inverse_transform = ~source.transform
            seen_offsets = set()
            for geometry, _ in target_crowns:
                center = geometry.centroid
                center_column, center_row = inverse_transform * (center.x, center.y)
                column_offset = int(min(max(round(center_column - tile_pixels / 2), 0), max(0, source.width - tile_pixels)))
                row_offset = int(min(max(round(center_row - tile_pixels / 2), 0), max(0, source.height - tile_pixels)))
                if (row_offset, column_offset) in seen_offsets:
                    continue
                seen_offsets.add((row_offset, column_offset))
                window = Window(column_offset, row_offset, min(tile_pixels, source.width - column_offset), min(tile_pixels, source.height - row_offset))
                try:
                    if (source.dataset_mask(window=window) == 0).mean() > config.maximum_nodata_fraction:
                        continue
                    lines = yolo_lines(target_crowns, window, source.transform, config.minimum_crown_fraction)
                    if not lines:
                        continue
                    key = f"{year}__{site}__{stem}__center_r{row_offset}_c{column_offset}"
                    save_tile(read_rgb(source, window), image_directory / f"{key}.jpg", config)
                    (label_directory / f"{key}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                    positives.append(_record(key, site, year, True, tif_path, other_crown_count=0))
                except Exception as error:
                    print(f"[WARNING] Center tile failed in {tif_path.name}: {error}")
    return positives, negative_candidates


def sample_negatives(candidates: list[dict], positive_count: int, config: DatasetConfig) -> list[dict]:
    """Take a fixed negative ratio, preferring tiles that contain other labelled crowns."""
    target_count = int(config.negative_to_positive_ratio * positive_count)
    ranked = sorted(candidates, key=lambda candidate: candidate["other_crown_count"], reverse=True)
    hard_count = min(round(config.hard_negative_fraction * target_count), len(ranked))
    hard, remaining = ranked[:hard_count], ranked[hard_count:]
    generator = random.Random(config.seed)
    generator.shuffle(remaining)
    return hard + remaining[: max(0, target_count - len(hard))]


def write_negatives(candidates: list[dict], image_directory: Path, label_directory: Path, config: DatasetConfig) -> list[dict]:
    """Materialize selected empty-label tiles, opening each source raster once."""
    records = []
    by_raster: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_raster[candidate["source_tif"]].append(candidate)
    for source_name, group in by_raster.items():
        with rasterio.open(source_name) as source:
            for candidate in group:
                image_path = image_directory / f"{candidate['key']}.jpg"
                label_path = label_directory / f"{candidate['key']}.txt"
                if not image_path.exists():
                    window = Window(candidate["column_offset"], candidate["row_offset"], candidate["width"], candidate["height"])
                    save_tile(read_rgb(source, window), image_path, config)
                label_path.touch(exist_ok=True)
                records.append({key: candidate[key] for key in ("key", "site", "year", "is_positive", "source_tif", "other_crown_count")})
    return records


def write_yolo_split(dataset_root: Path, name: str, train: list[dict], validation: list[dict]) -> None:
    """Write image lists and a portable Ultralytics dataset YAML file."""
    split_directory = dataset_root / "splits"
    split_directory.mkdir(exist_ok=True)
    for role, records in (("train", train), ("val", validation)):
        paths = [str((dataset_root / "images" / f"{record['key']}.jpg").resolve()) for record in records]
        (split_directory / f"{name}_{role}.txt").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
    (dataset_root / f"data_{name}.yaml").write_text(
        f"path: {dataset_root.resolve().as_posix()}\ntrain: splits/{name}_train.txt\nval: splits/{name}_val.txt\nnc: 1\nnames: ['shihuahuaco']\n",
        encoding="utf-8",
    )


def write_group_folds(dataset_root: Path, records: list[dict], fold_count: int = 5) -> None:
    """Create optional site-grouped folds and a 2023-to-2025 cross-year split."""
    tiles_per_site: dict[str, int] = defaultdict(int)
    for record in records:
        tiles_per_site[record["site"]] += 1
    fold_sizes = [0] * fold_count
    site_to_fold = {}
    for site, count in sorted(tiles_per_site.items(), key=lambda item: -item[1]):
        fold = min(range(fold_count), key=lambda index: fold_sizes[index])
        site_to_fold[site] = fold
        fold_sizes[fold] += count
    for fold in range(fold_count):
        write_yolo_split(dataset_root, f"fold{fold}", [record for record in records if site_to_fold[record["site"]] != fold], [record for record in records if site_to_fold[record["site"]] == fold])
    write_yolo_split(dataset_root, "crossyear", [record for record in records if record["year"] == 2023], [record for record in records if record["year"] == 2025])


def build_dataset(config: DatasetConfig, limit: int, fresh_state: bool) -> None:
    """Build all tile images, labels, manifest, and optional generic validation splits."""
    for path in (config.shapefile_2023, config.shapefile_2025, config.tif_root_2023, config.tif_root_2025):
        if not path.exists():
            raise FileNotFoundError(path)
    image_directory = config.output_dir / "images"
    label_directory = config.output_dir / "labels"
    state_directory = config.output_dir / "_state"
    image_directory.mkdir(parents=True, exist_ok=True)
    label_directory.mkdir(exist_ok=True)
    if fresh_state and state_directory.exists():
        shutil.rmtree(state_directory)
    state_directory.mkdir(exist_ok=True)
    (config.output_dir / "dataset_config_used.json").write_text(json.dumps(config.serializable(), indent=2), encoding="utf-8")

    targets_2023 = load_crowns(config.shapefile_2023, config.species_field, config.target_species_substring, True)
    others_2023 = load_crowns(config.shapefile_2023, config.species_field, config.target_species_substring, False)
    targets_2025 = load_crowns(config.shapefile_2025, config.species_field, config.target_species_substring, True)
    jobs = [(path, 2023, targets_2023, others_2023) for path in find_tiffs(config.tif_root_2023)]
    jobs += [(path, 2025, targets_2025, None) for path in find_tiffs(config.tif_root_2025)]
    if limit:
        jobs = jobs[:limit]
    all_positives, all_candidates = [], []
    for index, (tif_path, year, targets, others) in enumerate(jobs, start=1):
        cached = load_state(state_directory, tif_path)
        if cached is None:
            positives, candidates = process_tif(tif_path, year, targets, others, image_directory, label_directory, config)
            save_state(state_directory, tif_path, positives, candidates)
        else:
            positives, candidates = cached
        all_positives.extend(positives)
        all_candidates.extend(candidates)
        print(f"[{index}/{len(jobs)}] {tif_path.name}: {len(positives)} positive tiles")
    negatives = write_negatives(sample_negatives(all_candidates, len(all_positives), config), image_directory, label_directory, config)
    records = all_positives + negatives
    with (config.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "site", "year", "is_positive", "other_crown_count", "source_tif"])
        writer.writeheader()
        writer.writerows(records)
    write_group_folds(config.output_dir, records)
    print(f"Built {len(records)} tiles: {len(all_positives)} positive and {len(negatives)} negative.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the dataset YAML configuration.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rasters for a smoke test.")
    parser.add_argument("--fresh-state", action="store_true", help="Discard only resumable per-raster state before rebuilding.")
    args = parser.parse_args()
    build_dataset(load_dataset_config(args.config), args.limit, args.fresh_state)


if __name__ == "__main__":
    main()

