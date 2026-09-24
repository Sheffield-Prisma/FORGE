"""Evaluate all eight post-hoc OSR scoring methods for a saved classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import load_config
from .constants import LABEL_MAP
from .data import feature_loader, known_loaders, unknown_loader
from .model import DINOv2Classifier
from .scoring import extract_outputs, oscr, osr_auc, score_all_methods


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[DINOv2Classifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DINOv2Classifier(num_classes=len(LABEL_MAP)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _evaluate_split(
    model: DINOv2Classifier,
    train_features: np.ndarray,
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    config,
    checkpoint: dict,
    years: list[int] | None,
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    _, _, known_test_loader = known_loaders(
        config.dataset_root, train_years=checkpoint["train_years"], test_years=years,
        image_size=config.image_size, batch_size=config.batch_size,
        num_workers=config.num_workers, seed=config.seed, weighted_sampling=False,
    )
    unknown_test_loader = unknown_loader(
        config.dataset_root, years=years, classes=None, image_size=config.image_size,
        batch_size=config.batch_size, num_workers=config.num_workers, seed=config.seed,
    )
    known_features, known_logits, known_labels = extract_outputs(model, known_test_loader, device)
    unknown_features, unknown_logits, _ = extract_outputs(model, unknown_test_loader, device)
    predicted_labels = known_logits.argmax(axis=1)
    known_is_correct = predicted_labels == known_labels
    score_pairs = score_all_methods(
        train_features, train_logits, train_labels, known_features, known_logits, unknown_features, unknown_logits,
        model_head=model.head, device=device, knn_k=config.knn_k,
        covariance_regularization=config.covariance_regularization,
    )
    methods = {
        name: {"auroc": osr_auc(known_scores, unknown_scores), "oscr": oscr(known_scores, known_is_correct, unknown_scores)}
        for name, (known_scores, unknown_scores) in score_pairs.items()
    }
    result = {
        "test_years": years or [2023, 2025],
        "closed_set_accuracy": float(accuracy_score(known_labels, predicted_labels)),
        "closed_set_macro_f1": float(f1_score(known_labels, predicted_labels, average="macro")),
        "known_test_samples": int(len(known_labels)),
        "unknown_test_samples": int(len(unknown_features)),
        "methods": methods,
    }
    arrays = {f"{name}_known": pair[0] for name, pair in score_pairs.items()}
    arrays.update({f"{name}_unknown": pair[1] for name, pair in score_pairs.items()})
    arrays.update({"known_labels": known_labels, "known_predictions": predicted_labels})
    return result, arrays


def evaluate(checkpoint_path: Path, config_path: str | Path, eval_years: list[int], device: torch.device) -> Path:
    """Run combined and per-year evaluations, saving JSON metrics and score arrays."""
    config = load_config(config_path)
    model, checkpoint = _load_model(checkpoint_path, device)
    if tuple(checkpoint["known_classes"]) != config.known_classes:
        raise ValueError("The checkpoint known-class order does not match the configuration.")
    train_loader = feature_loader(
        config.dataset_root, train_years=checkpoint["train_years"], image_size=config.image_size,
        batch_size=config.batch_size, num_workers=config.num_workers, seed=config.seed,
    )
    train_features, train_logits, train_labels = extract_outputs(model, train_loader, device)
    output_directory = checkpoint_path.parent / "evaluation"
    output_directory.mkdir(exist_ok=True)

    report = {"checkpoint": str(checkpoint_path), "train_years": checkpoint["train_years"], "results": {}}
    evaluation_sets: list[tuple[str, list[int] | None]] = [("combined", None)]
    evaluation_sets.extend((str(year), [year]) for year in eval_years)
    for name, years in evaluation_sets:
        result, arrays = _evaluate_split(
            model, train_features, train_logits, train_labels, config, checkpoint, years, device
        )
        report["results"][name] = result
        np.savez_compressed(output_directory / f"scores_{name}.npz", **arrays)
        print(
            f"{name}: accuracy={result['closed_set_accuracy']:.3f}, "
            f"macro-F1={result['closed_set_macro_f1']:.3f}, "
            f"best AUROC={max(result['methods'].items(), key=lambda pair: pair[1]['auroc'])[0]}"
        )
    report_path = output_directory / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved metrics to {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-years", type=int, nargs="+", default=[2023, 2025])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config, args.eval_years, torch.device(args.device))


if __name__ == "__main__":
    main()

