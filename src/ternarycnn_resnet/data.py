"""Dataset configuration and data-loader helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DatasetName = Literal["mnist", "cifar10", "cifar100"]


@dataclass(frozen=True)
class DatasetSpec:
    """Image and label dimensions needed to construct a classifier."""

    channels: int
    classes: int
    image_size: int


DATASET_SPECS: dict[DatasetName, DatasetSpec] = {
    "mnist": DatasetSpec(channels=1, classes=10, image_size=28),
    "cifar10": DatasetSpec(channels=3, classes=10, image_size=32),
    "cifar100": DatasetSpec(channels=3, classes=100, image_size=32),
}

_NORMALIZATION = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def limit_dataset(dataset: Dataset, size: int, seed: int) -> Dataset:
    """Return a deterministic random subset, or the original dataset for size zero."""
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def _transforms(name: DatasetName) -> tuple[transforms.Compose, transforms.Compose]:
    mean, std = _NORMALIZATION[name]
    normalize = transforms.Normalize(mean, std)
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if name == "mnist":
        return test_transform, test_transform
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, test_transform


def _datasets(
    name: DatasetName,
    data_dir: Path,
    *,
    fake_data: bool,
) -> tuple[Dataset, Dataset]:
    spec = DATASET_SPECS[name]
    train_transform, test_transform = _transforms(name)
    if fake_data:
        image_size = (spec.channels, spec.image_size, spec.image_size)
        return (
            datasets.FakeData(
                size=128,
                image_size=image_size,
                num_classes=spec.classes,
                transform=train_transform,
            ),
            datasets.FakeData(
                size=64,
                image_size=image_size,
                num_classes=spec.classes,
                transform=test_transform,
            ),
        )

    dataset_class = {
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }[name]
    return (
        dataset_class(data_dir, train=True, download=True, transform=train_transform),
        dataset_class(data_dir, train=False, download=True, transform=test_transform),
    )


def create_loaders(
    name: DatasetName,
    *,
    data_dir: Path,
    batch_size: int,
    train_samples: int,
    test_samples: int,
    workers: int,
    seed: int,
    device: torch.device,
    fake_data: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create loaders for MNIST, CIFAR-10, or CIFAR-100."""
    train_data, test_data = _datasets(name, data_dir, fake_data=fake_data)
    train_data = limit_dataset(train_data, train_samples, seed)
    test_data = limit_dataset(test_data, test_samples, seed)
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(train_data, shuffle=True, **options),
        DataLoader(test_data, shuffle=False, **options),
    )
