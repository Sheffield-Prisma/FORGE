"""Create the qualitative look-alike figure referenced in the report draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .config import load_config

PAIRS = (
    ("CASTANHEIRA", "MANITE", "Hard: AUROC 0.61"),
    ("MACARANDUBA", "MANITE", "Hard: AUROC 0.62"),
    ("TAUARI", "GARAPA", "Hard: AUROC 0.70"),
    ("CEDRO", "PINHO", "Easy: AUROC 0.92"),
)


def _sample_paths(dataset_root: Path, manifest: pd.DataFrame, class_name: str, split: str, count: int) -> list[Path]:
    candidates = manifest[(manifest["class"] == class_name) & (manifest["split"] == split)]
    sampled = candidates.sample(n=min(count, len(candidates)), random_state=42)
    return [dataset_root / relative_path for relative_path in sampled["relative_path"]]


def make_lookalike_figure(config_path: str | Path, output: Path, examples_per_class: int) -> Path:
    """Plot selected hard/easy unknown species next to their nearest known classes."""
    config = load_config(config_path)
    known = pd.read_csv(config.dataset_root / "metadata_known.csv")
    unknown = pd.read_csv(config.dataset_root / "metadata_unknown.csv")
    figure, axes = plt.subplots(
        len(PAIRS), examples_per_class * 2,
        figsize=(examples_per_class * 3, len(PAIRS) * 2.1),
        squeeze=False,
    )
    for row, (unknown_class, known_class, label) in enumerate(PAIRS):
        unknown_paths = _sample_paths(config.dataset_root, unknown, unknown_class, "test_unknown", examples_per_class)
        known_paths = _sample_paths(config.dataset_root, known, known_class, "test", examples_per_class)
        for column in range(examples_per_class * 2):
            axis = axes[row, column]
            axis.axis("off")
            paths = unknown_paths if column < examples_per_class else known_paths
            path_index = column % examples_per_class
            if path_index < len(paths):
                with Image.open(paths[path_index]) as image:
                    axis.imshow(image.convert("RGB"))
            if column == 0:
                axis.set_title(f"{unknown_class}\nunknown", color="#c2410c", fontsize=9)
            if column == examples_per_class:
                axis.set_title(f"{known_class}\nknown", color="#1d4ed8", fontsize=9)
        axes[row, 0].set_ylabel(label, rotation=90, labelpad=32, fontsize=9)
    figure.suptitle("Unknown (left) and visually similar known crops (right)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples-per-class", type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or config.artifacts_dir / "lookalike_pairs.png"
    make_lookalike_figure(args.config, output, args.examples_per_class)


if __name__ == "__main__":
    main()

