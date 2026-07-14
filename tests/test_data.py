from pathlib import Path

import pytest
import torch

from ternarycnn_resnet.data import DATASET_SPECS, create_loaders


@pytest.mark.parametrize("name", ["mnist", "cifar10", "cifar100"])
def test_fake_dataset_shapes_and_labels(name: str) -> None:
    spec = DATASET_SPECS[name]
    train_loader, test_loader = create_loaders(
        name,
        data_dir=Path("unused"),
        batch_size=4,
        train_samples=8,
        test_samples=4,
        workers=0,
        seed=7,
        device=torch.device("cpu"),
        fake_data=True,
    )
    images, labels = next(iter(train_loader))
    assert images.shape == (4, spec.channels, spec.image_size, spec.image_size)
    assert labels.shape == (4,)
    assert labels.min() >= 0
    assert labels.max() < spec.classes
    assert len(test_loader.dataset) == 4
