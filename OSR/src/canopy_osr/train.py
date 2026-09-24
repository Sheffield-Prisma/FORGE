"""Two-stage closed-set and SupCon model training."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as functional

from .config import ExperimentConfig, load_config
from .constants import LABEL_MAP
from .data import known_loaders
from .model import DINOv2Classifier


class ProjectionHead(nn.Module):
    """The 768 -> 512 -> 128 projection head used only by SupCon training."""

    def __init__(self, input_dimension: int = 768, output_dimension: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, 512), nn.ReLU(inplace=True), nn.Linear(512, output_dimension)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.layers(features), dim=1)


def supervised_contrastive_loss(
    features: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Compute the supervised contrastive loss, excluding anchors with no positive."""
    similarities = features @ features.T / temperature
    similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    same_class = labels[:, None].eq(labels[None, :])
    not_self = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = same_class & not_self
    log_denominator = torch.log((torch.exp(similarities) * not_self).sum(dim=1, keepdim=True) + 1e-12)
    log_probability = similarities - log_denominator
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.sum() * 0.0
    mean_positive_log_probability = (log_probability * positives).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_positive_log_probability[valid].mean()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def _validation_macro_f1(model: DINOv2Classifier, loader, device: torch.device) -> float:
    model.eval()
    predictions, labels = [], []
    for images, batch_labels in loader:
        predictions.extend(model(images.to(device, non_blocking=True)).argmax(dim=1).cpu().tolist())
        labels.extend(batch_labels.tolist())
    return float(f1_score(labels, predictions, average="macro"))


def _run_stage(
    model: DINOv2Classifier,
    projection: ProjectionHead | None,
    train_loader,
    validation_loader,
    *,
    epochs: int,
    learning_rate: float,
    patience_limit: int,
    supcon_weight: float,
    temperature: float,
    device: torch.device,
    stage_name: str,
) -> tuple[float, list[dict[str, float]]]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if projection is not None:
        trainable.extend(projection.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    cross_entropy = nn.CrossEntropyLoss()
    best_f1, best_state, stale_epochs = -float("inf"), None, 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        if projection is not None:
            projection.train()
        total_loss, total_examples = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            features = model.forward_features(images)
            logits = model.head(features)
            loss = cross_entropy(logits, labels)
            if projection is not None:
                loss = loss + supcon_weight * supervised_contrastive_loss(
                    projection(features), labels, temperature
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_examples += len(labels)

        validation_f1 = _validation_macro_f1(model, validation_loader, device)
        record = {"stage": stage_name, "epoch": epoch, "loss": total_loss / total_examples, "validation_macro_f1": validation_f1}
        history.append(record)
        print(f"{stage_name} epoch {epoch:02d}: loss={record['loss']:.4f}, validation macro-F1={validation_f1:.4f}")
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience_limit:
                print(f"{stage_name}: early stopping after {epoch} epochs.")
                break
    if best_state is None:
        raise RuntimeError(f"{stage_name} did not complete an epoch.")
    model.load_state_dict(best_state)
    return best_f1, history


def train(config: ExperimentConfig, train_years: list[int], method: str, device: torch.device) -> Path:
    """Train a CE-only or CE+SupCon classifier and return its checkpoint path."""
    _seed_everything(config.seed)
    train_loader, validation_loader, _ = known_loaders(
        config.dataset_root, train_years=train_years, test_years=None,
        image_size=config.image_size, batch_size=config.batch_size,
        num_workers=config.num_workers, seed=config.seed,
    )
    model = DINOv2Classifier(num_classes=len(LABEL_MAP)).to(device)
    stage1_f1, history = _run_stage(
        model, None, train_loader, validation_loader, epochs=config.stage1_epochs,
        learning_rate=config.stage1_learning_rate, patience_limit=config.early_stopping_patience,
        supcon_weight=0.0, temperature=config.supcon_temperature, device=device, stage_name="stage1",
    )
    model.unfreeze_last_blocks(config.unfrozen_blocks)
    projection = ProjectionHead().to(device) if method == "supcon" else None
    stage2_f1, stage2_history = _run_stage(
        model, projection, train_loader, validation_loader, epochs=config.stage2_epochs,
        learning_rate=config.stage2_learning_rate, patience_limit=config.early_stopping_patience,
        supcon_weight=config.supcon_weight, temperature=config.supcon_temperature,
        device=device, stage_name="stage2",
    )
    history.extend(stage2_history)

    run_name = f"{method}_train{'_'.join(map(str, train_years))}"
    run_directory = config.artifacts_dir / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = run_directory / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "method": method,
            "train_years": train_years,
            "known_classes": list(config.known_classes),
            "best_validation_macro_f1": {"stage1": stage1_f1, "stage2": stage2_f1},
        },
        checkpoint,
    )
    (run_directory / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_directory / "config_used.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved checkpoint to {checkpoint}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["baseline", "supcon"], default="baseline")
    parser.add_argument("--train-years", type=int, nargs="+", default=[2023, 2025])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(load_config(args.config), args.train_years, args.method, torch.device(args.device))


if __name__ == "__main__":
    main()

