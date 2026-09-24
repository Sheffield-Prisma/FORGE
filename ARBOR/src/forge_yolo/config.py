"""Configuration loading for the reproducible dataset builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Parameters that determine the exact set of emitted training tiles."""

    output_dir: Path
    shapefile_2023: Path
    shapefile_2025: Path
    tif_root_2023: Path
    tif_root_2025: Path
    species_field: str = "NOMBRE_CIE"
    target_species_substring: str = "Dipteryx"
    tile_size_m: float = 56.0
    overlap: float = 0.5
    output_pixels: int = 1024
    maximum_nodata_fraction: float = 0.30
    minimum_crown_fraction: float = 0.50
    negative_exclusion_fraction: float = 0.05
    include_centered_tiles: bool = True
    negative_to_positive_ratio: float = 3.0
    hard_negative_fraction: float = 0.70
    jpeg_quality: int = 95
    seed: int = 42

    def validate(self) -> "DatasetConfig":
        if self.tile_size_m <= 0 or self.output_pixels <= 0:
            raise ValueError("tile_size_m and output_pixels must be positive.")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1).")
        if not 0 <= self.minimum_crown_fraction <= 1:
            raise ValueError("minimum_crown_fraction must be in [0, 1].")
        if not 0 <= self.negative_exclusion_fraction < self.minimum_crown_fraction:
            raise ValueError("negative_exclusion_fraction must be below minimum_crown_fraction.")
        if self.negative_to_positive_ratio < 0:
            raise ValueError("negative_to_positive_ratio must not be negative.")
        return self

    def serializable(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Read YAML and fail if an accidental or misspelled key is present."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(DatasetConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")
    for key in ("output_dir", "shapefile_2023", "shapefile_2025", "tif_root_2023", "tif_root_2025"):
        if key in raw:
            raw[key] = Path(raw[key])
    return DatasetConfig(**raw).validate()

