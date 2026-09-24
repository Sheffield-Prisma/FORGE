"""Fine-tune YOLO11m-seg on the single-class Shihuahuaco dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def metric_summary(metrics) -> dict[str, float | None]:
    """Read segmentation metrics defensively across Ultralytics releases."""
    segmentation = getattr(metrics, "seg", None)
    boxes = getattr(metrics, "box", None)
    def value(namespace, name):
        metric = getattr(namespace, name, None)
        return float(metric) if metric is not None else None
    return {
        "mask_map50": value(segmentation, "map50"),
        "mask_map50_95": value(segmentation, "map"),
        "box_map50": value(boxes, "map50"),
        "box_map50_95": value(boxes, "map"),
    }


def train(args: argparse.Namespace) -> Path:
    """Run the reported augmentation and training protocol, then revalidate best.pt."""
    data_path = Path(args.data).resolve()
    project = Path(args.project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings["data"] = str(data_path)
    settings["project"] = str(project)
    (project / f"{args.name}_training_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    model = YOLO(args.model)
    model.train(
        data=str(data_path), epochs=args.epochs, imgsz=args.image_size, batch=args.batch,
        device=args.device, workers=args.workers, project=str(project), name=args.name,
        patience=args.patience, optimizer="auto", cos_lr=True, seed=args.seed, amp=args.amp,
        degrees=180.0, flipud=0.5, fliplr=0.5, scale=0.5, translate=0.1,
        mosaic=1.0, close_mosaic=10, copy_paste=0.3, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        plots=True, val=True,
    )
    run_directory = Path(model.trainer.save_dir)
    best_checkpoint = run_directory / "weights" / "best.pt"
    best_model = YOLO(str(best_checkpoint))
    metrics = best_model.val(data=str(data_path), imgsz=args.image_size, device=args.device, split="val", plots=True)
    summary_path = run_directory / "validation_summary.json"
    summary_path.write_text(json.dumps(metric_summary(metrics), indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Validation summary: {summary_path}")
    return best_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML, such as data_abl_both.yaml.")
    parser.add_argument("--model", default="yolo11m-seg.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name", default="shihuahuaco_seg")
    parser.add_argument("--project", default="runs")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
