"""Train a ternary ResNet on MNIST."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import torch
from torch import nn

from .checkpoints import save_ternary_checkpoint
from .data import DATASET_SPECS, create_loaders
from .engine import evaluate_model, select_device, train_one_epoch
from .models import build_resnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("r18", "r50", "r101"), default="r18")
    parser.add_argument("--dataset", choices=("mnist", "cifar10", "cifar100"), default="mnist")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache/ternarycnn")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=16, help="First-stage channel count")
    parser.add_argument("--train-samples", type=int, default=12_000, help="Use 0 for all")
    parser.add_argument("--test-samples", type=int, default=2_000, help="Use 0 for all")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=Path, help="Optional output path for the state dict")
    parser.add_argument(
        "--save-ternary",
        type=Path,
        help="Optional output path for frozen int8 ternary weights and scales",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one tiny epoch on generated images without downloading MNIST",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.width = min(args.width, 8)
        args.workers = 0
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_loader, test_loader = create_loaders(
        args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        workers=args.workers,
        seed=args.seed,
        device=device,
        fake_data=args.smoke_test,
    )
    spec = DATASET_SPECS[args.dataset]
    model = build_resnet(
        args.model,
        num_classes=spec.classes,
        in_channels=spec.channels,
        width=args.width,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} dataset={args.dataset} model={args.model} parameters={parameters:,}")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        scheduler.step()
        print(
            f"epoch={epoch:02d}/{args.epochs:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.2%} "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.2%}"
        )
    print(f"completed in {time.perf_counter() - started:.1f}s")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"saved {args.save}")
    if args.save_ternary:
        save_ternary_checkpoint(
            model,
            args.save_ternary,
            model_name=args.model,
            dataset=args.dataset,
            width=args.width,
        )
        print(f"saved ternary weights {args.save_ternary}")


if __name__ == "__main__":
    main()
