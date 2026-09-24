# FORGE

This repository groups three related research modules for analysing Amazonian tree crowns in UAV orthomosaics.

| Module            | Purpose                                                      |
| ----------------- | ------------------------------------------------------------ |
| `ARBOR`           | Single-class YOLO-seg detection and segmentation of Shihuahuaco (*Dipteryx*) crowns. Includes dataset construction, training, and held-out validation. |
| `crown_retrieval` | DINOv2 feature extraction and cosine-similarity retrieval for georeferenced crown candidates. Builds a reusable feature index and exports ranked results for QGIS. |
| `OSR`             | Multi-species open-set recognition research. Trains a DINOv2 classifier for five known species and evaluates rejection of six held-out unknown species. |

