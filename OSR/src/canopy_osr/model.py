"""DINOv2 ViT-B/14 classifier used in all reported experiments."""

from __future__ import annotations

import torch
from torch import nn


class DINOv2Classifier(nn.Module):
    """A DINOv2 backbone with the documented 768 -> 256 -> C classifier head."""

    feature_dimension = 768

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True, verbose=False
        )
        self.head = nn.Sequential(
            nn.Linear(self.feature_dimension, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze every backbone parameter for the first training stage."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_blocks(self, block_count: int) -> None:
        """Fine-tune only the final transformer blocks and final normalization."""
        if block_count < 1 or block_count > len(self.backbone.blocks):
            raise ValueError(f"block_count must be between 1 and {len(self.backbone.blocks)}")
        self.freeze_backbone()
        for block in self.backbone.blocks[-block_count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad = True

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the trainable CLS-token representation."""
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))

