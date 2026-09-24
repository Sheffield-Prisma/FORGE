# Crown Retrieval

**DINOv2 cosine-similarity image retrieval** for tree crowns predicted by YOLO segmentation.

Workflow: given an orthomosaic and a georeferenced YOLO candidate layer, it extracts one embedding per polygon-masked crown, saves a durable index, and uses that index to retrieve visually similar crowns.

## Scope

Included:

- DINOv2 extraction from RGB polygon crops with a neutral-gray masked background;
- persistent feature indexing for a raster/candidate layer pair;
- FID/source-ID based Top-K cosine-similarity query and GeoPackage export;
- labeled retrieval evaluation with mAP, Top-K precision/recall/F1, and optional monthly metrics;
- class-similarity heatmaps and t-SNE visualization.

## Install

Use Python 3.10+ and a CUDA-enabled PyTorch installation for efficient extraction.

```bash
pip install -e .
```

The first run downloads the selected DINOv2 backbone through `torch.hub`. Generated indexes, results, rasters, model weights, and GeoPackages are ignored by Git.

## 1. Build the feature index

The candidate layer should contain georeferenced crown polygons produced by YOLO. Its CRS is aligned to the raster before every crop is extracted, which avoids using geometry coordinates in the wrong projection.

```bash
python -m crown_retrieval.build_index \
  --raster /path/to/orthomosaic.tif \
  --candidates /path/to/yolo_crowns.gpkg \
  --output indexes/plot_2025_isa
```

If the candidate layer has a stable identifier other than `FID`, specify it explicitly:

```bash
python -m crown_retrieval.build_index \
  --raster /path/to/orthomosaic.tif \
  --candidates /path/to/yolo_crowns.gpkg \
  --id-column crown_id \
  --output indexes/plot_2025_isa
```

The index contains:

```text
indexes/plot_2025_isa/
├── embeddings.npy        # L2-normalized DINOv2 vectors; one row per candidate
├── candidates.gpkg       # aligned polygons plus retrieval_id and source_id
├── index_metadata.json   # model and extraction settings
└── failures.csv          # candidates that could not be cropped/embedded
```

The default extractor is `dinov2_vits14`, 224-pixel input, and gray background value 124, matching the supplied Stage 2 script. Use `--model dinov2_vitb14` for a larger backbone. Rebuild an existing index only with explicit `--overwrite`.

## 2. Retrieve similar crowns

After indexing, no raster reads or model inference are needed for a query:

```bash
python -m crown_retrieval.query \
  --index indexes/plot_2025_isa \
  --query-id 37 \
  --top-k 100
```

`--query-id` targets the original candidate `source_id` by default. This is the detected `FID` field when present; otherwise it is the original layer row index. The output is a sorted GeoPackage with `rank` and `similarity` fields, ready to load in QGIS.

To query using the internal index row instead, pass `--id-column retrieval_id`. Use `--top-k 0` to export every non-query candidate.

## 3. Evaluate labeled retrieval

Evaluation needs a CSV whose ID values correspond to `source_id` and whose labels identify the true species. The default expected columns are `id` and `class_id`.

```bash
python -m crown_retrieval.evaluate \
  --index indexes/labeled_crowns \
  --annotations /path/to/annotations.csv \
  --annotation-id-column id \
  --label-column class_id \
  --max-k 100
```

Outputs include overall mean AP, macro class mAP, per-query AP, per-class best retrieval depth/F1, and a class-mAP plot. Add `--month-column month` when the annotation CSV already has a month field to also write monthly mAP.

The program joins labels by ID rather than assuming the CSV and `.npy` happen to have the same row order.

## 4. Visualize the feature space

```bash
python -m crown_retrieval.visualize \
  --index indexes/labeled_crowns \
  --annotations /path/to/annotations.csv \
  --annotation-id-column id \
  --label-column class_id \
  --min-samples 20
```

This creates a mean ± standard-deviation class cosine-similarity heatmap and a global t-SNE embedding plot. Classes below `--min-samples` are excluded from visualization only; they are not removed from the index.



