"""Checkpoint helpers for trainable and frozen ternary ResNets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from ternarylayers import Bit
from torch import nn

TERNARY_CHECKPOINT_FORMAT = "ternarycnn-resnet-ternary-v1"
_TRAINING_LAYER_TYPES = (Bit.Conv2d, Bit.ConvTranspose2d, Bit.Linear)


@torch.no_grad()
def freeze_ternary_weights(model: nn.Module) -> nn.Module:
    """Clone a model and replace every QAT layer with its frozen inference layer."""
    frozen = deepcopy(model).cpu().eval()

    def convert_children(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, _TRAINING_LAYER_TYPES):
                setattr(module, name, child.to_ternary())
            else:
                convert_children(child)

    convert_children(frozen)
    return frozen


def save_ternary_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    model_name: str,
    dataset: str,
    width: int,
) -> None:
    """Save frozen int8 ternary weights, scales, and model metadata."""
    frozen = freeze_ternary_weights(model)
    payload = {
        "format": TERNARY_CHECKPOINT_FORMAT,
        "model": model_name,
        "dataset": dataset,
        "width": width,
        "state_dict": frozen.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def is_ternary_checkpoint(payload: Any) -> bool:
    """Return whether a loaded object is a supported frozen ternary checkpoint."""
    return isinstance(payload, dict) and payload.get("format") == TERNARY_CHECKPOINT_FORMAT

