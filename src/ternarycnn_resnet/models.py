"""Model construction utilities."""

from __future__ import annotations

from typing import Literal

from ternarylayers import ResNetModels
from torch import nn

ModelName = Literal["r18", "r50", "r101"]

_BUILDERS = {
    "r18": ResNetModels.R18,
    "r50": ResNetModels.R50,
    "r101": ResNetModels.R101,
}


def build_resnet(
    name: ModelName = "r18",
    *,
    num_classes: int = 10,
    in_channels: int = 1,
    width: int = 16,
    small_stem: bool = True,
    scale_op: str = "mean",
) -> nn.Module:
    """Build a ternary ResNet from a compact, stable interface."""
    if name not in _BUILDERS:
        choices = ", ".join(_BUILDERS)
        raise ValueError(f"unknown model {name!r}; expected one of: {choices}")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if in_channels <= 0:
        raise ValueError("in_channels must be positive")
    if width <= 0:
        raise ValueError("width must be positive")

    return _BUILDERS[name](
        num_classes=num_classes,
        in_ch=in_channels,
        inplanes=width,
        small_stem=small_stem,
        scale_op=scale_op,
    ).build()
