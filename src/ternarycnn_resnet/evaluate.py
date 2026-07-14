"""Evaluate a saved ternary ResNet checkpoint on MNIST."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn

from .checkpoints import freeze_ternary_weights, is_ternary_checkpoint
from .data import DATASET_SPECS, create_loaders
from .engine import evaluate_model, select_device
from .models import build_resnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model", choices=("r18", "r50", "r101"))
    parser.add_argument("--dataset", choices=("mnist", "cifar10", "cifar100"))
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache/ternarycnn")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=0, help="Use 0 for all")
    parser.add_argument("--width", type=int)
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ternary = is_ternary_checkpoint(payload)
    metadata = payload if ternary else {}
    model_name = args.model or metadata.get("model", "r18")
    dataset = args.dataset or metadata.get("dataset", "mnist")
    width = args.width or metadata.get("width", 16)

    spec = DATASET_SPECS[dataset]
    model = build_resnet(
        model_name,
        num_classes=spec.classes,
        in_channels=spec.channels,
        width=width,
    )
    if ternary:
        model = freeze_ternary_weights(model)
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict)
    model.to(device)

    _, test_loader = create_loaders(
        dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_samples=1,
        test_samples=args.test_samples,
        workers=args.workers,
        seed=args.seed,
        device=device,
    )
    loss, accuracy = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
    print(
        f"dataset={dataset} model={model_name} ternary={ternary} "
        f"samples={len(test_loader.dataset):,} "
        f"loss={loss:.4f} accuracy={accuracy:.2%}"
    )


if __name__ == "__main__":
    main()
