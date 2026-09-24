"""Loading and validating experiment configuration files."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from .constants import KNOWN_CLASSES, UNKNOWN_CLASSES


@dataclass(frozen=True)
class ExperimentConfig:
    """All settings that affect dataset construction or reported metrics."""

    dataset_root: Path
    artifacts_dir: Path = Path("artifacts")
    known_classes: tuple[str, ...] = KNOWN_CLASSES
    unknown_classes: tuple[str, ...] = UNKNOWN_CLASSES
    seed: int = 42
    image_size: int = 224
    known_confidence: float = 0.45
    unknown_confidence: float = 0.55
    test_tif_fraction: float = 0.20
    train_cap_per_class_year: int | None = 300
    unknown_test_cap_per_class_year: int | None = 200
    crop_padding_ratio: float = 0.20
    batch_size: int = 32
    num_workers: int = 4
    stage1_epochs: int = 25
    stage1_learning_rate: float = 1e-3
    stage2_epochs: int = 20
    stage2_learning_rate: float = 1e-5
    unfrozen_blocks: int = 4
    early_stopping_patience: int = 10
    supcon_weight: float = 0.5
    supcon_temperature: float = 0.1
    knn_k: int = 5
    covariance_regularization: float = 1e-6

    def validate(self) -> "ExperimentConfig":
        if not 0 < self.test_tif_fraction < 1:
            raise ValueError("test_tif_fraction must be between 0 and 1.")
        if set(self.known_classes) & set(self.unknown_classes):
            raise ValueError("Known and unknown class lists must not overlap.")
        if self.known_classes != KNOWN_CLASSES or self.unknown_classes != UNKNOWN_CLASSES:
            raise ValueError(
                "This repository reproduces one fixed benchmark; keep the documented "
                "known and unknown class lists (including known-class order)."
            )
        if self.image_size <= 0 or self.batch_size <= 0:
            raise ValueError("image_size and batch_size must be positive.")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML configuration and reject unknown keys early."""
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")

    for key in ("dataset_root", "artifacts_dir"):
        if key in raw:
            raw[key] = Path(raw[key])
    for key in ("known_classes", "unknown_classes"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return ExperimentConfig(**raw).validate()
