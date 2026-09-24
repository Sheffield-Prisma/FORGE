"""Supporting analyses reported for the Amazon canopy OSR study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import load_config
from .constants import KNOWN_CLASSES, LABEL_MAP, UNKNOWN_CLASSES
from .data import feature_loader, known_loaders, unknown_loader
from .model import DINOv2Classifier
from .scoring import MahalanobisScorer, extract_outputs, osr_auc


def _checkpoint_model(path: Path, device: torch.device) -> tuple[DINOv2Classifier, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DINOv2Classifier(len(LABEL_MAP)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def per_class_unknown(config_path: str | Path, checkpoint_path: Path, device: torch.device) -> Path:
    """Measure unknown-class difficulty and nearest known centroid assignments."""
    config = load_config(config_path)
    model, checkpoint = _checkpoint_model(checkpoint_path, device)
    train_features, _, train_labels = extract_outputs(
        model,
        feature_loader(config.dataset_root, train_years=checkpoint["train_years"], image_size=config.image_size,
                       batch_size=config.batch_size, num_workers=config.num_workers, seed=config.seed),
        device,
    )
    scorer = MahalanobisScorer.fit(train_features, train_labels, config.covariance_regularization)
    _, _, known_loader = known_loaders(
        config.dataset_root, train_years=checkpoint["train_years"], test_years=None,
        image_size=config.image_size, batch_size=config.batch_size, num_workers=config.num_workers,
        seed=config.seed, weighted_sampling=False,
    )
    known_features, _, _ = extract_outputs(model, known_loader, device)
    known_scores = scorer.score(known_features)

    results = {}
    for class_name in config.unknown_classes:
        loader = unknown_loader(
            config.dataset_root, years=None, classes=[class_name], image_size=config.image_size,
            batch_size=config.batch_size, num_workers=config.num_workers, seed=config.seed,
        )
        unknown_features, _, _ = extract_outputs(model, loader, device)
        nearest = scorer.nearest_class_index(unknown_features)
        counts = np.bincount(nearest, minlength=len(KNOWN_CLASSES))
        nearest_index = int(counts.argmax())
        results[class_name] = {
            "samples": int(len(unknown_features)),
            "auroc": osr_auc(known_scores, scorer.score(unknown_features)),
            "nearest_known_class": config.known_classes[nearest_index],
            "nearest_known_fraction": float(counts[nearest_index] / counts.sum()),
        }
    output = checkpoint_path.parent / "analysis"
    output.mkdir(exist_ok=True)
    report_path = output / "per_class_unknown.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved per-class analysis to {report_path}")
    return report_path


def _test_features(model, config, year: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    _, _, loader = known_loaders(
        config.dataset_root, train_years=[2023, 2025], test_years=[year], image_size=config.image_size,
        batch_size=config.batch_size, num_workers=config.num_workers, seed=config.seed, weighted_sampling=False,
    )
    features, _, labels = extract_outputs(model, loader, device)
    return features, labels


def domain_gap(
    config_path: str | Path, checkpoint_path: Path | None, raw_dinov2: bool, device: torch.device
) -> Path:
    """Quantify 2023/2025 feature-domain separability and centroid displacement."""
    config = load_config(config_path)
    if raw_dinov2:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True, verbose=False).to(device)

        class RawFeatureModel(torch.nn.Module):
            def __init__(self, raw_backbone):
                super().__init__()
                self.backbone = raw_backbone
                self.head = torch.nn.Identity()

            def forward_features(self, images):
                return self.backbone(images)

        model = RawFeatureModel(backbone).to(device).eval()
        output = config.artifacts_dir / "raw_dinov2_domain_gap"
    else:
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required unless --raw-dinov2 is selected.")
        model, _ = _checkpoint_model(checkpoint_path, device)
        output = checkpoint_path.parent / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    features_2023, labels_2023 = _test_features(model, config, 2023, device)
    features_2025, labels_2025 = _test_features(model, config, 2025, device)
    features = np.concatenate([features_2023, features_2025])
    domain_labels = np.r_[np.zeros(len(features_2023)), np.ones(len(features_2025))]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    domain_auc = cross_val_score(
        LogisticRegression(max_iter=2000), features, domain_labels, cv=folds, scoring="roc_auc"
    ).mean()
    centroids_2023 = np.stack([features_2023[labels_2023 == index].mean(axis=0) for index in range(len(LABEL_MAP))])
    centroids_2025 = np.stack([features_2025[labels_2025 == index].mean(axis=0) for index in range(len(LABEL_MAP))])
    cross_year = np.linalg.norm(centroids_2023 - centroids_2025, axis=1)
    between = [
        np.linalg.norm(centroids_2023[left] - centroids_2023[right])
        for left in range(len(LABEL_MAP)) for right in range(left + 1, len(LABEL_MAP))
    ]
    report = {
        "feature_extractor": "raw_dinov2" if raw_dinov2 else "fine_tuned_classifier",
        "domain_auc": float(domain_auc),
        "mean_cross_year_centroid_shift": float(cross_year.mean()),
        "mean_between_class_centroid_distance": float(np.mean(between)),
        "cross_year_to_between_class_ratio": float(cross_year.mean() / np.mean(between)),
        "per_class_cross_year_shift": {class_name: float(cross_year[index]) for index, class_name in enumerate(config.known_classes)},
    }
    report_path = output / "domain_gap.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved domain-gap analysis to {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    per_class = commands.add_parser("per-class", help="Analyze unknown species separately.")
    per_class.add_argument("--config", required=True)
    per_class.add_argument("--checkpoint", type=Path, required=True)
    per_class.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    domain = commands.add_parser("domain-gap", help="Measure the 2023/2025 feature-domain gap.")
    domain.add_argument("--config", required=True)
    domain.add_argument("--checkpoint", type=Path)
    domain.add_argument("--raw-dinov2", action="store_true")
    domain.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.command == "per-class":
        per_class_unknown(args.config, args.checkpoint, device)
    else:
        domain_gap(args.config, args.checkpoint, args.raw_dinov2, device)


if __name__ == "__main__":
    main()
