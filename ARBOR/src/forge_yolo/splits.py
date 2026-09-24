"""Create held-out ablation and in-sample deployment splits from a tile manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def resolve_dataset_root(dataset: str | None, manifest: str | None) -> tuple[Path, Path]:
    """Accept either a dataset directory or an explicit manifest path."""
    if manifest:
        manifest_path = Path(manifest).resolve()
        return manifest_path.parent, manifest_path
    if dataset:
        dataset_root = Path(dataset).resolve()
        return dataset_root, dataset_root / "manifest.csv"
    raise ValueError("Provide --dataset or --manifest.")


def write_split(dataset_root: Path, name: str, train_keys: list[str], validation_keys: list[str]) -> Path:
    """Write Ultralytics image lists and the corresponding one-class YAML file."""
    split_directory = dataset_root / "splits"
    split_directory.mkdir(exist_ok=True)
    for role, keys in (("train", train_keys), ("val", validation_keys)):
        paths = [str((dataset_root / "images" / f"{key}.jpg").resolve()) for key in keys]
        (split_directory / f"{name}_{role}.txt").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
    yaml_path = dataset_root / f"data_{name}.yaml"
    yaml_path.write_text(
        f"path: {dataset_root.as_posix()}\ntrain: splits/{name}_train.txt\nval: splits/{name}_val.txt\nnc: 1\nnames: ['shihuahuaco']\n",
        encoding="utf-8",
    )
    return yaml_path


def counts(frame: pd.DataFrame, keys: list[str]) -> dict[str, int]:
    """Report the positive/negative composition of a split."""
    subset = frame[frame["key"].isin(keys)]
    positives = int((subset["is_positive"] == 1).sum())
    return {"positive": positives, "negative": int(len(subset) - positives), "total": int(len(subset))}


def make_ablation_split(dataset_root: Path, manifest: Path, test_sites: list[str], test_fraction: float) -> dict:
    """Hold out fixed 2023 sites to measure the contribution of 2025 data."""
    frame = pd.read_csv(manifest)
    frame["key"] = frame["key"].astype(str)
    data_2023 = frame[frame["year"] == 2023]
    data_2025 = frame[frame["year"] == 2025]
    positive_by_site = data_2023[data_2023["is_positive"] == 1].groupby("site").size().sort_values(ascending=False)
    if positive_by_site.empty:
        raise ValueError("The manifest has no positive 2023 tiles.")
    if not test_sites:
        target = test_fraction * int(positive_by_site.sum())
        total, test_sites = 0, []
        for site, positive_count in positive_by_site.iloc[1:].items():
            test_sites.append(str(site))
            total += int(positive_count)
            if total >= target and len(test_sites) >= 2:
                break
    unknown_sites = sorted(set(test_sites) - set(positive_by_site.index))
    if unknown_sites:
        raise ValueError(f"Requested held-out site(s) are absent from 2023 positives: {unknown_sites}")
    test_site_set = set(test_sites)
    test_keys = data_2023[data_2023["site"].isin(test_site_set)]["key"].tolist()
    train_2023_keys = data_2023[~data_2023["site"].isin(test_site_set)]["key"].tolist()
    train_both_keys = train_2023_keys + data_2025["key"].tolist()
    output = {
        "held_out_2023_sites": sorted(test_site_set),
        "test": counts(frame, test_keys),
        "train_2023only": counts(frame, train_2023_keys),
        "train_both": counts(frame, train_both_keys),
        "yaml_files": {
            "2023only": str(write_split(dataset_root, "abl_2023only", train_2023_keys, test_keys)),
            "both": str(write_split(dataset_root, "abl_both", train_both_keys, test_keys)),
        },
    }
    (dataset_root / "ablation_split.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def make_deployment_split(dataset_root: Path, manifest: Path, validation_fraction: float, seed: int) -> dict:
    """Stratify a random monitoring split for the all-data deployment model.

    This split shares sites with the training set and must not be reported as a
    generalization metric. Use the ablation's held-out 2023 sites for that.
    """
    frame = pd.read_csv(manifest)
    frame["key"] = frame["key"].astype(str)
    generator = random.Random(seed)
    train_keys, validation_keys = [], []
    for _, group in frame.groupby("is_positive"):
        keys = group["key"].tolist()
        generator.shuffle(keys)
        validation_count = max(1, round(len(keys) * validation_fraction))
        validation_keys.extend(keys[:validation_count])
        train_keys.extend(keys[validation_count:])
    generator.shuffle(train_keys)
    generator.shuffle(validation_keys)
    yaml_path = write_split(dataset_root, "deploy", train_keys, validation_keys)
    output = {
        "purpose": "in-sample training monitoring only; not a generalization estimate",
        "validation_fraction": validation_fraction,
        "seed": seed,
        "train": counts(frame, train_keys),
        "validation": counts(frame, validation_keys),
        "yaml_file": str(yaml_path),
    }
    (dataset_root / "deployment_split.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ablation = commands.add_parser("ablation", help="Create the fixed held-out 2023 ablation split.")
    deploy = commands.add_parser("deploy", help="Create the in-sample deployment monitoring split.")
    for command in (ablation, deploy):
        command.add_argument("--dataset")
        command.add_argument("--manifest")
    ablation.add_argument("--test-sites", default="", help="Comma-separated 2023 sites; default selects them deterministically.")
    ablation.add_argument("--test-fraction", type=float, default=0.20)
    deploy.add_argument("--validation-fraction", type=float, default=0.10)
    deploy.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dataset_root, manifest = resolve_dataset_root(args.dataset, args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    if args.command == "ablation":
        site_list = [site.strip() for site in args.test_sites.split(",") if site.strip()]
        result = make_ablation_split(dataset_root, manifest, site_list, args.test_fraction)
    else:
        result = make_deployment_split(dataset_root, manifest, args.validation_fraction, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

