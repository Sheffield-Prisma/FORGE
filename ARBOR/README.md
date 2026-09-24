# Forge: Shihuahuaco YOLO-seg

Detecting and segmenting **Shihuahuaco (*Dipteryx*) tree crowns** in Amazon UAV orthomosaics with a single-class YOLO11m-seg model.

Workflow:

1. geospatial annotation-to-tile dataset construction;
2. fixed held-out ablation and deployment monitoring splits;
3. YOLO11m-seg training;
4. tile-level mAP validation and optional crown-level validation from georeferenced predictions.



## Study protocol

| Component | Value |
| --- | --- |
| Target class | `Dipteryx` crowns, class index 0, name `shihuahuaco` |
| Ground tile size | 56 m |
| Tile overlap | 50% (28 m stride) |
| Model input | 1024 × 1024 pixels |
| Positive-label rule | At least 50% of a crown must be visible in the tile |
| Partial-crown exclusion | Tiles exposing 5–50% of a target crown are discarded |
| Centered positive tiles | One deduplicated crown-centred tile per target crown |
| Negative ratio | 3 negatives per positive; 70% selected from tiles containing other annotated crowns |
| Model | YOLO11m-seg, initialized from COCO-pretrained weights |
| Training | 150 epochs, patience 30, AMP enabled unless it becomes unstable |



## Installation

Use Python 3.10+ and an environment with a compatible CUDA PyTorch build for GPU training.

```bash
pip install -e .
```

Ultralytics downloads the starting `yolo11m-seg.pt` weight on first use unless it is already cached. Raw imagery, shapefiles, generated datasets, run outputs, weights, and GeoPackages are excluded from version control.

## 1. Build the dataset

Copy the configuration template and replace every path:

```bash
cp configs/dataset.example.yaml configs/dataset.local.yaml
python -m forge_yolo.build_dataset --config configs/dataset.local.yaml
```

For a small smoke test, add `--limit 4`. Re-run the same command after interruption to resume from per-raster state. `--fresh-state` removes only that resumable state cache; use it when changing any tile-generation parameter.

The output directory contains:

```text
Shihuahuaco_v2/
├── images/                         # 1024-pixel RGB JPEG tiles
├── labels/                         # matching YOLO segmentation labels
├── manifest.csv                    # tile, site, year, class-presence, source-TIF metadata
├── splits/                         # image lists used by Ultralytics
├── data_fold*.yaml                 # optional site-grouped folds
├── data_crossyear.yaml             # optional 2023 train / 2025 validation split
└── dataset_config_used.json         # immutable record of the inputs and parameters
```

## 2. Create reproducible validation splits

The report uses a controlled ablation: two fixed 2023 sites are held out, while the only difference between the models is whether 2025 tiles are added to training.

```bash
python -m forge_yolo.splits ablation \
  --dataset /path/to/Shihuahuaco_v2 \
  --test-sites "16-CONP-MAD-SD-023-15,GOREMAD-GRRNYGA-DRFFSDFFS-TAHP-MAD-D-008-15"
```

This produces:

- `data_abl_2023only.yaml`: non-held-out 2023 tiles for training;
- `data_abl_both.yaml`: the same 2023 training tiles plus all 2025 tiles;
- the same fixed held-out 2023 tiles for validation in both cases.

If site names differ from the archived data naming convention, omit `--test-sites` for a deterministic automatic selection, then inspect `ablation_split.json` before training.

For the final deployment model, use all data with a random, stratified monitoring validation set:

```bash
python -m forge_yolo.splits deploy --dataset /path/to/Shihuahuaco_v2
```

`data_deploy.yaml` is for checkpoint selection only. Its validation tiles share sites with training and are **not** a generalization metric.

## 3. Train

```bash
# Controlled 2023-only ablation
python -m forge_yolo.train \
  --data /path/to/Shihuahuaco_v2/data_abl_2023only.yaml \
  --name abl_2023only --project runs

# Controlled 2023 + 2025 ablation
python -m forge_yolo.train \
  --data /path/to/Shihuahuaco_v2/data_abl_both.yaml \
  --name abl_both_v2 --project runs

# Final all-data deployment model
python -m forge_yolo.train \
  --data /path/to/Shihuahuaco_v2/data_deploy.yaml \
  --name deploy_final --project runs
```

Training uses the aerial-image augmentations from the report: full rotation, vertical/horizontal flips, scale/translation, mosaic, and segmentation copy-paste. Each run writes its exact arguments and a post-training `validation_summary.json` next to `weights/best.pt`.

If the first epochs show NaN segmentation loss, restart the run with `--no-amp`.

For Slurm, adapt [scripts/train_slurm.sh](scripts/train_slurm.sh).

## 4. Validate a saved checkpoint

Run an independent tile-level validation whenever comparing two checkpoints. Both models must use the **same YAML/test tile list**.

```bash
python -m forge_yolo.validate \
  --checkpoint runs/abl_both_v2/weights/best.pt \
  --data /path/to/Shihuahuaco_v2/data_abl_both.yaml
```

The saved JSON contains mask mAP50, mask mAP50–95, and box equivalents. The report's externally reportable benchmark is the held-out 2023 ablation, not the optimistic `deploy_final` monitoring score.

For crown-level validation on held-out plots, supply already georeferenced prediction files and their source TIFs:

```bash
python -m forge_yolo.crown_metrics \
  --predictions heldout_a.gpkg heldout_b.gpkg \
  --rasters heldout_a.tif heldout_b.tif \
  --ground-truth /path/to/copas_2023_condatos_vs2.shp \
  --output crown_metrics.json
```

This computes one-to-one polygon matches and combined precision, recall, and F1. It does not perform inference; the report's terminal inference implementation is intentionally out of scope.



