from __future__ import annotations

import pytest
import torch

from ternarycnn_resnet import build_resnet


@pytest.mark.parametrize("name", ["r18", "r50", "r101"])
def test_resnet_output_shape(name: str) -> None:
    model = build_resnet(name, num_classes=7, in_channels=1, width=4).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 1, 32, 32))
    assert output.shape == (2, 7)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("num_classes", 0, "num_classes must be positive"),
        ("in_channels", 0, "in_channels must be positive"),
        ("width", 0, "width must be positive"),
    ],
)
def test_resnet_rejects_invalid_dimensions(keyword: str, value: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_resnet(**{keyword: value})


def test_resnet_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_resnet("r34")
