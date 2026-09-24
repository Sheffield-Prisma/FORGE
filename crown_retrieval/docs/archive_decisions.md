# Archive triage decisions

The supplied Stage 2 archive contains three scripts, all related to tree-crown feature retrieval. They were reorganized rather than copied unchanged.

| Original file | New module(s) | Change |
| --- | --- | --- |
| `crown_retrieval.py` | `embeddings.py`, `build_index.py`, `query.py` | Split feature extraction from query execution. Features are now cached once per candidate layer instead of recomputed for every query. Candidate and raster CRS are aligned before both query and gallery crops. |
| `evaluation.py` | `evaluate.py` | Replaced order-dependent inputs with an ID-based label join; preserves AP, class mAP, precision/recall, F1, and optional temporal summary. |
| `tsne.py` | `visualize.py` | Retains class similarity (mean ± standard deviation) and global t-SNE, while removing hard-coded filenames, fixed flight-to-month mappings, and commented-out plotting branches. |

All implementation comments and command-line documentation are in English. No YOLO detector, raster inference, or unrelated model-training code was added because this repository begins with the Stage 1 candidate polygons.

