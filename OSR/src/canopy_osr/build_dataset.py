"""Scan NetFlora detections and construct the leak-free crop dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from .config import ExperimentConfig, load_config


def _require_data_libraries():
    """Import optional geospatial dependencies only for data-construction commands."""
    try:
        import cv2
        import rasterio
        from rasterio.windows import Window
    except ImportError as error:
        raise RuntimeError("Install the data extras with: pip install -e '.[data]'") from error
    return cv2, rasterio, Window


class NetFloraDetector:
    """Thin adapter around the external NetFlora YOLOv5 repository.

    NetFlora's code and weights are external research dependencies. They are
    intentionally referenced by command-line paths rather than copied into
    this repository.
    """

    def __init__(self, repository: Path, weights: Path, image_size: int, device: str = "") -> None:
        if not repository.exists() or not weights.exists():
            raise FileNotFoundError("Both --netflora-repo and --weights must exist.")
        sys.path.insert(0, str(repository))
        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_coords
        from utils.torch_utils import select_device

        import torch

        self.torch = torch
        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.device = select_device(device)
        self.model = attempt_load(str(weights), map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.image_size = check_img_size(image_size, s=self.stride)
        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        if self.device.type != "cpu":
            self.model.half()

    def predict(self, bgr_image: np.ndarray, confidence: float, iou: float) -> list[tuple[str, float, int, int, int, int]]:
        image = self.letterbox(bgr_image, self.image_size, stride=self.stride)[0]
        image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        tensor = self.torch.from_numpy(image).to(self.device)
        tensor = tensor.half() if self.device.type != "cpu" else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        predictions = self.non_max_suppression(self.model(tensor, augment=False)[0], confidence, iou)
        output = []
        for detections in predictions:
            if not len(detections):
                continue
            detections[:, :4] = self.scale_coords(tensor.shape[2:], detections[:, :4], bgr_image.shape).round()
            for *xyxy, score, class_index in detections:
                class_name = self.names[int(class_index)].split("_")[0].upper()
                x1, y1, x2, y2 = (int(value) for value in xyxy)
                output.append((class_name, float(score), x1, y1, x2, y2))
        return output


def _parse_sources(entries: Iterable[str]) -> list[tuple[int, Path]]:
    """Parse repeatable ``YEAR=PATH`` source specifications."""
    sources = []
    for entry in entries:
        try:
            year_text, directory = entry.split("=", maxsplit=1)
            sources.append((int(year_text), Path(directory)))
        except ValueError as error:
            raise ValueError(f"Invalid source '{entry}'; use YEAR=PATH.") from error
    return sources


def _tile_starts(length: int, tile_size: int, margin_ratio: float, stride: int) -> list[int]:
    margin = int(length * margin_ratio)
    final_start = length - margin - tile_size
    if final_start < margin:
        return []
    starts = list(range(margin, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _scan_tif(
    detector: NetFloraDetector,
    tif_path: Path,
    year: int,
    *,
    tile_size: int,
    stride: int,
    margin_ratio: float,
    min_valid_ratio: float,
    confidence: float,
    iou: float,
) -> list[dict]:
    cv2, rasterio, Window = _require_data_libraries()
    records: list[dict] = []
    with rasterio.open(tif_path) as source:
        xs = _tile_starts(source.width, tile_size, margin_ratio, stride)
        ys = _tile_starts(source.height, tile_size, margin_ratio, stride)
        for y_offset in ys:
            for x_offset in xs:
                window = Window(x_offset, y_offset, tile_size, tile_size)
                data = source.read(window=window)
                if source.count >= 4 and (data[3] > 0).mean() < min_valid_ratio:
                    continue
                rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for class_name, score, x1, y1, x2, y2 in detector.predict(bgr, confidence, iou):
                    center_x = x_offset + (x1 + x2) / 2
                    center_y = y_offset + (y1 + y2) / 2
                    longitude, latitude = source.transform * (center_x, center_y)
                    records.append(
                        {
                            "year": year,
                            "tif": tif_path.name,
                            "source_path": str(tif_path.resolve()),
                            "class": class_name,
                            "confidence": round(score, 6),
                            "center_x_geo": longitude,
                            "center_y_geo": latitude,
                            "tile_x": x_offset,
                            "tile_y": y_offset,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )
    return records


def scan_detections(args: argparse.Namespace) -> None:
    """Scan all requested imagery and save all NetFlora classes to one manifest."""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detector = NetFloraDetector(Path(args.netflora_repo), Path(args.weights), args.tile_size, args.device)
    image_paths = []
    for year, directory in _parse_sources(args.source):
        if not directory.exists():
            raise FileNotFoundError(f"Source directory does not exist: {directory}")
        image_paths.extend((year, path) for extension in ("*.tif", "*.tiff") for path in directory.rglob(extension))
    image_paths.sort(key=lambda item: (item[0], str(item[1])))
    if not image_paths:
        raise ValueError("No TIF/TIFF files were found in the supplied source directories.")

    records: list[dict] = []
    for index, (year, tif_path) in enumerate(image_paths, start=1):
        try:
            added = _scan_tif(
                detector, tif_path, year, tile_size=args.tile_size, stride=args.stride,
                margin_ratio=args.margin_ratio, min_valid_ratio=args.min_valid_ratio,
                confidence=args.confidence, iou=args.iou,
            )
            records.extend(added)
            print(f"[{index}/{len(image_paths)}] {year} {tif_path.name}: {len(added)} detections")
        except Exception as error:  # Continue scanning while making failures visible.
            print(f"[WARNING] Could not scan {tif_path}: {error}")
        if index % args.checkpoint_every == 0:
            pd.DataFrame(records).to_csv(output, index=False)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"Saved {len(records)} detections to {output}")


def _deduplicate(frame: pd.DataFrame, distance_meters: float) -> pd.DataFrame:
    """Keep the highest-confidence detection in each nearby same-class pair."""
    accepted_indices: list[int] = []
    for _, group in frame.groupby(["source_path", "class"], sort=False):
        accepted: list[tuple[float, float]] = []
        for index, row in group.sort_values("confidence", ascending=False).iterrows():
            point = (float(row["center_x_geo"]), float(row["center_y_geo"]))
            if all(np.hypot(point[0] - x, point[1] - y) >= distance_meters for x, y in accepted):
                accepted.append(point)
                accepted_indices.append(index)
    return frame.loc[accepted_indices].copy().reset_index(drop=True)


def _assign_tif_splits(frame: pd.DataFrame, fraction: float, seed: int) -> dict[str, str]:
    """Assign each source TIF to train/test within acquisition year."""
    assignments: dict[str, str] = {}
    for year, group in frame.groupby("year", sort=True):
        sources = sorted(group["source_path"].unique())
        random.Random(seed + int(year)).shuffle(sources)
        test_count = max(1, int(len(sources) * fraction))
        assignments.update({source: "test" if source in sources[:test_count] else "train" for source in sources})
    return assignments


def _crop(source, record: pd.Series, image_size: int, padding_ratio: float) -> np.ndarray | None:
    _, _, Window = _require_data_libraries()
    x1 = int(record["tile_x"]) + int(record["x1"])
    y1 = int(record["tile_y"]) + int(record["y1"])
    x2 = int(record["tile_x"]) + int(record["x2"])
    y2 = int(record["tile_y"]) + int(record["y2"])
    width, height = x2 - x1, y2 - y1
    if width < 16 or height < 16:
        return None
    side = int(max(width, height) * (1 + padding_ratio))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    left, top = max(0, center_x - side // 2), max(0, center_y - side // 2)
    right, bottom = min(source.width, center_x + side // 2), min(source.height, center_y + side // 2)
    if right - left < 8 or bottom - top < 8:
        return None
    data = source.read([1, 2, 3], window=Window(left, top, right - left, bottom - top))
    crop = np.transpose(data, (1, 2, 0)).clip(0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(crop, mode="RGB").resize((image_size, image_size), Image.Resampling.BILINEAR))


def _cap_groups(frame: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None:
        return frame
    groups = []
    for (_, _), group in frame.groupby(["class", "year"], sort=True):
        groups.append(group.sample(n=min(len(group), cap), random_state=seed))
    return pd.concat(groups, ignore_index=True) if groups else frame.iloc[0:0].copy()


def build_dataset(args: argparse.Namespace) -> None:
    """Create known and unknown crop manifests from all-class detections."""
    _, rasterio, _ = _require_data_libraries()
    config = load_config(args.config)
    detections_path = Path(args.detections)
    required_columns = {
        "year", "tif", "source_path", "class", "confidence", "center_x_geo", "center_y_geo",
        "tile_x", "tile_y", "x1", "y1", "x2", "y2",
    }
    detections = pd.read_csv(detections_path)
    missing = required_columns - set(detections.columns)
    if missing:
        raise ValueError(f"Detection file is missing columns: {', '.join(sorted(missing))}")

    selected = detections[
        (detections["class"].isin(config.known_classes) & (detections["confidence"] >= config.known_confidence))
        | (detections["class"].isin(config.unknown_classes) & (detections["confidence"] >= config.unknown_confidence))
    ].copy()
    deduplicated = _deduplicate(selected, args.dedup_distance)
    config.dataset_root.mkdir(parents=True, exist_ok=True)
    deduplicated.to_csv(config.dataset_root / "detections_deduplicated.csv", index=False)

    tif_split = _assign_tif_splits(deduplicated, config.test_tif_fraction, config.seed)
    known = deduplicated[deduplicated["class"].isin(config.known_classes)].copy()
    known["split"] = known["source_path"].map(tif_split)
    known_train = _cap_groups(known[known["split"] == "train"], config.train_cap_per_class_year, config.seed)
    known_test = known[known["split"] == "test"]
    known = pd.concat([known_train, known_test], ignore_index=True)

    # Unknown crops use only the same held-out TIFs as known test crops.
    unknown = deduplicated[deduplicated["class"].isin(config.unknown_classes)].copy()
    unknown = unknown[unknown["source_path"].map(tif_split) == "test"].copy()
    unknown = _cap_groups(unknown, config.unknown_test_cap_per_class_year, config.seed)
    unknown["split"] = "test_unknown"
    if known.empty:
        raise ValueError("No known detections remain after thresholding, splitting, and caps.")
    if unknown.empty:
        raise ValueError("No unknown detections remain on held-out TIFs after thresholding and caps.")

    manifests: dict[str, list[dict]] = {"known": [], "unknown": []}
    failed = defaultdict(int)
    combined = pd.concat([known.assign(kind="known"), unknown.assign(kind="unknown")], ignore_index=True)
    for source_path, group in combined.groupby("source_path", sort=True):
        with rasterio.open(source_path) as source:
            for row_index, record in group.iterrows():
                crop = _crop(source, record, config.image_size, config.crop_padding_ratio)
                if crop is None:
                    failed["crop"] += 1
                    continue
                split, class_name = str(record["split"]), str(record["class"])
                file_name = f"{class_name}_{int(record['year'])}_{Path(source_path).stem}_{row_index:06d}.png"
                relative_path = Path(split) / class_name / file_name
                destination = config.dataset_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop, mode="RGB").save(destination)
                manifests[str(record["kind"])].append(
                    {
                        "class": class_name,
                        "year": int(record["year"]),
                        "tif": record["tif"],
                        "split": split,
                        "confidence": record["confidence"],
                        "source_path": source_path,
                        "relative_path": relative_path.as_posix(),
                    }
                )

    for kind, records in manifests.items():
        pd.DataFrame(records).to_csv(config.dataset_root / f"metadata_{kind}.csv", index=False)
    known_counts = pd.DataFrame(manifests["known"]).groupby(["split", "class", "year"]).size()
    unknown_counts = pd.DataFrame(manifests["unknown"]).groupby(["class", "year"]).size()
    summary = {
        "known_crops": len(manifests["known"]),
        "unknown_crops": len(manifests["unknown"]),
        "failed_crops": failed["crop"],
        "known_by_split_class_year": {"/".join(map(str, key)): int(value) for key, value in known_counts.items()},
        "unknown_by_class_year": {"/".join(map(str, key)): int(value) for key, value in unknown_counts.items()},
    }
    (config.dataset_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Created {summary['known_crops']} known and {summary['unknown_crops']} unknown crops at {config.dataset_root}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="Run NetFlora over source TIFs and write all detections.")
    scan.add_argument("--netflora-repo", required=True)
    scan.add_argument("--weights", required=True)
    scan.add_argument("--source", action="append", required=True, metavar="YEAR=PATH")
    scan.add_argument("--output", required=True)
    scan.add_argument("--tile-size", type=int, default=1024)
    scan.add_argument("--stride", type=int, default=1024)
    scan.add_argument("--margin-ratio", type=float, default=0.05)
    scan.add_argument("--min-valid-ratio", type=float, default=0.50)
    scan.add_argument("--confidence", type=float, default=0.45)
    scan.add_argument("--iou", type=float, default=0.45)
    scan.add_argument("--checkpoint-every", type=int, default=20)
    scan.add_argument("--device", default="")
    scan.set_defaults(function=scan_detections)
    build = commands.add_parser("build", help="Deduplicate, split, and crop the OSR dataset.")
    build.add_argument("--config", required=True)
    build.add_argument("--detections", required=True)
    build.add_argument("--dedup-distance", type=float, default=3.0)
    build.set_defaults(function=build_dataset)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
