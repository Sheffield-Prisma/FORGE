"""Dataset access and deterministic train/validation partitioning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .constants import IMAGENET_MEAN, IMAGENET_STD, LABEL_MAP, UNKNOWN_LABEL

Role = Literal["train", "validation", "test_known", "test_unknown"]


def image_transform(training: bool, image_size: int) -> transforms.Compose:
    """Return the augmentation policy documented for the OSR experiments."""
    common = [
        transforms.Resize((image_size, image_size)),
    ]
    if training:
        common.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            ]
        )
    common.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(common)


class CrownDataset(Dataset):
    """Load crop records from the standardized dataset manifest.

    The builder creates two manifests at ``dataset_root``:
    ``metadata_known.csv`` for known train/test crops and
    ``metadata_unknown.csv`` for held-out unknown crops.  Each contains a
    ``relative_path`` column, so code never relies on a workstation-specific
    absolute path.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        role: Role,
        *,
        years: Iterable[int] | None = None,
        classes: Iterable[str] | None = None,
        image_size: int = 224,
        training_transform: bool = False,
        validation_fraction: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.role = role
        self.transform = image_transform(training_transform, image_size)
        manifest = (
            self.dataset_root / "metadata_unknown.csv"
            if role == "test_unknown"
            else self.dataset_root / "metadata_known.csv"
        )
        if not manifest.exists():
            raise FileNotFoundError(f"Dataset manifest was not found: {manifest}")
        frame = pd.read_csv(manifest)

        if years is not None:
            frame = frame[frame["year"].isin(list(years))]
        if classes is not None:
            frame = frame[frame["class"].isin(list(classes))]

        if role in {"train", "validation"}:
            frame = frame[frame["split"] == "train"].copy()
            frame = self._take_validation_partition(
                frame, role == "validation", validation_fraction, seed
            )
            frame["label"] = frame["class"].map(LABEL_MAP)
        elif role == "test_known":
            frame = frame[frame["split"] == "test"].copy()
            frame["label"] = frame["class"].map(LABEL_MAP)
        elif role == "test_unknown":
            frame = frame[frame["split"] == "test_unknown"].copy()
            frame["label"] = UNKNOWN_LABEL
        else:
            raise ValueError(f"Unsupported role: {role}")

        if frame.empty:
            raise ValueError(f"No records available for role={role}, years={years}.")
        self.frame = frame.reset_index(drop=True)
        self.labels = self.frame["label"].astype(int).to_numpy()

    @staticmethod
    def _take_validation_partition(
        frame: pd.DataFrame,
        keep_validation: bool,
        fraction: float,
        seed: int,
    ) -> pd.DataFrame:
        """Split each class/year cell reproducibly, without image leakage."""
        validation_indices: list[int] = []
        for (_, _), group in frame.groupby(["class", "year"], sort=True):
            count = max(1, round(len(group) * fraction))
            rng = np.random.default_rng(seed + int(group["year"].iloc[0]))
            chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
            validation_indices.extend(chosen.tolist())
        is_validation = frame.index.isin(validation_indices)
        return frame.loc[is_validation if keep_validation else ~is_validation].copy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.frame.iloc[index]
        image_path = self.dataset_root / record["relative_path"]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        label = torch.tensor(int(record["label"]), dtype=torch.long)
        return tensor, label


def _loader(dataset: Dataset, *, batch_size: int, num_workers: int, sampler=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def known_loaders(
    dataset_root: str | Path,
    *,
    train_years: Iterable[int],
    test_years: Iterable[int] | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    weighted_sampling: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build training, validation, and known-class test loaders."""
    train_set = CrownDataset(
        dataset_root, "train", years=train_years, image_size=image_size,
        training_transform=True, seed=seed,
    )
    validation_set = CrownDataset(
        dataset_root, "validation", years=train_years, image_size=image_size,
        seed=seed,
    )
    test_set = CrownDataset(
        dataset_root, "test_known", years=test_years, image_size=image_size,
        seed=seed,
    )

    sampler = None
    if weighted_sampling:
        counts = np.bincount(train_set.labels, minlength=len(LABEL_MAP)).astype(float)
        class_weights = 1.0 / np.maximum(counts, 1.0)
        sample_weights = torch.as_tensor(class_weights[train_set.labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return (
        _loader(train_set, batch_size=batch_size, num_workers=num_workers, sampler=sampler),
        _loader(validation_set, batch_size=batch_size, num_workers=num_workers),
        _loader(test_set, batch_size=batch_size, num_workers=num_workers),
    )


def feature_loader(
    dataset_root: str | Path,
    *,
    train_years: Iterable[int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Return unaugmented known training crops for fitting OSR scorers."""
    dataset = CrownDataset(
        dataset_root, "train", years=train_years, image_size=image_size, seed=seed
    )
    return _loader(dataset, batch_size=batch_size, num_workers=num_workers)


def unknown_loader(
    dataset_root: str | Path,
    *,
    years: Iterable[int] | None,
    classes: Iterable[str] | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Return held-out unknown crops, optionally restricted by year or species."""
    dataset = CrownDataset(
        dataset_root, "test_unknown", years=years, classes=classes,
        image_size=image_size, seed=seed,
    )
    return _loader(dataset, batch_size=batch_size, num_workers=num_workers)

