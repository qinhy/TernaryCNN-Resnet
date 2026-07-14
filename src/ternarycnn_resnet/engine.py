"""Reusable training and evaluation loops."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def select_device(requested: str) -> torch.device:
    """Resolve an explicit device name or choose the best available accelerator."""
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return mean loss and accuracy."""
    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        count = targets.size(0)
        loss_sum += loss.item() * count
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += count
    return loss_sum / seen, correct / seen


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate a model and return mean loss and accuracy."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        count = targets.size(0)
        loss_sum += criterion(logits, targets).item() * count
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += count
    return loss_sum / seen, correct / seen
