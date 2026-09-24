"""Run a repeatable Ultralytics validation for a trained YOLO-seg checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from .train import metric_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    model = YOLO(str(args.checkpoint))
    metrics = model.val(data=str(args.data), imgsz=args.image_size, device=args.device, split=args.split, plots=True)
    output = args.output or args.checkpoint.parent.parent / f"{args.split}_metrics.json"
    output.write_text(json.dumps(metric_summary(metrics), indent=2), encoding="utf-8")
    print(f"Saved validation metrics to {output}")


if __name__ == "__main__":
    main()

