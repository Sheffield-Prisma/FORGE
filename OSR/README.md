# Amazon Canopy OSR

**open-set recognition (OSR) of Amazon tree crowns** in UAV orthophotos. It implements the experiment described in `docs/report_draft.md`:

- five known classes: COPAIBA, CUMARU, GARAPA, MANITE, PINHO;
- six held-out unknown classes: CAUCHO, CASTANHEIRA, CEDRO, MACARANDUBA, TAUARI, TAXI;
- DINOv2 ViT-B/14 with two-stage fine-tuning;
- optional supervised contrastive learning (SupCon);
- six post-hoc OSR scores: MSP, MaxLogit, Energy, Entropy, Mahalanobis, KNN;
- closed-set accuracy / macro-F1, AUROC, OSCR, per-unknown analysis.



## Install

Use Python 3.10 or later. Create an isolated environment, then install the package and the data/plotting dependencies:

```bash
pip install -e ".[data,plots]"
```

The DINOv2 backbone is loaded through `torch.hub` on first use. The dataset scan additionally requires a local checkout of the NetFlora YOLOv5 code compatible with its weights; its path is supplied at run time and is not vendored here.



## Workflow

### 1. Scan all source imagery once

The scan stores **all NetFlora classes** in a single CSV. That avoids the archived inconsistency where the known- and unknown-class builders consumed different scan outputs.

```bash
python -m canopy_osr.build_dataset scan \
  --netflora-repo /path/to/netflora-yolov5 \
  --weights /path/to/PMFS_Embrapa00.pt \
  --source 2023=/path/to/2023-imagery \
  --source 2025=/path/to/2025-imagery \
  --output /path/to/detections_all.csv
```

Only TIF/TIFF files are considered. The command records the exact source path with every detection, so crop extraction is not vulnerable to duplicate TIF filenames in different directories. It checkpoints the CSV during long scans.

### 2. Build the standardized crop dataset

```bash
python -m canopy_osr.build_dataset build \
  --config configs/local.yaml \
  --detections /path/to/detections_all.csv
```

This command:

1. applies class-specific confidence thresholds and geographic de-duplication;
2. splits source TIFs by year before cropping;
3. caps only the training cells, while keeping known test crops intact;
4. draws unknown crops only from held-out test TIFs, preventing tile-level context leakage;
5. saves images and two portable CSV manifests under `dataset_root`.

Expected layout:

```text
dataset_root/
├── train/<known class>/*.png
├── test/<known class>/*.png
├── test_unknown/<unknown class>/*.png
├── metadata_known.csv
├── metadata_unknown.csv
├── detections_deduplicated.csv
└── dataset_summary.json
```

### 3. Train a baseline or SupCon model

```bash
# Two-stage cross-entropy baseline
python -m canopy_osr.train --config configs/local.yaml --method baseline

# Same setup, adding SupCon in stage 2
python -m canopy_osr.train --config configs/local.yaml --method supcon
```

Pass `--train-years 2023` or `--train-years 2025` for a single-year training condition. Each run saves a model checkpoint, exact configuration, and epoch-level history to `artifacts_dir/<method>_train<years>/`.

### 4. Evaluate every post-hoc OSR method

```bash
python -m canopy_osr.evaluate \
  --config configs/local.yaml \
  --checkpoint artifacts/baseline_train2023_2025/model.pt
```

The command evaluates the combined test set and each requested year. It saves `metrics.json` and compressed score arrays below the checkpoint. Scores are always oriented so a larger value means “more known-like.”

### 5. Run supporting analyses

```bash
# Per-unknown-class AUROC and most-confused known class
python -m canopy_osr.analyze per-class \
  --config configs/local.yaml \
  --checkpoint artifacts/supcon_train2023_2025/model.pt

# Domain AUC and cross-year / between-class centroid ratio
python -m canopy_osr.analyze domain-gap \
  --config configs/local.yaml \
  --checkpoint artifacts/baseline_train2023_2025/model.pt

# Repeat domain-gap analysis with frozen, raw DINOv2 features
python -m canopy_osr.analyze domain-gap \
  --config configs/local.yaml --raw-dinov2
```

Create the report's qualitative hard/easy look-alike figure with:

```bash
python -m canopy_osr.visualize --config configs/local.yaml
```

