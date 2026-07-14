from pathlib import Path

import torch
from ternarylayers import Bit

from ternarycnn_resnet import build_resnet, freeze_ternary_weights, save_ternary_checkpoint
from ternarycnn_resnet.checkpoints import TERNARY_CHECKPOINT_FORMAT, is_ternary_checkpoint


def test_freeze_ternary_weights_preserves_output_and_freezes_every_layer() -> None:
    torch.manual_seed(7)
    model = build_resnet("r18", in_channels=1, width=2).eval()
    sample = torch.randn(2, 1, 28, 28)

    frozen = freeze_ternary_weights(model)

    assert any(isinstance(module, Bit.Conv2dInfer) for module in frozen.modules())
    assert any(isinstance(module, Bit.LinearInfer) for module in frozen.modules())
    assert not any(isinstance(module, Bit.Conv2d) for module in frozen.modules())
    assert not any(isinstance(module, Bit.Linear) for module in frozen.modules())
    with torch.inference_mode():
        torch.testing.assert_close(frozen(sample), model(sample), rtol=1e-5, atol=1e-6)


def test_ternary_checkpoint_round_trip(tmp_path: Path) -> None:
    model = build_resnet("r18", in_channels=1, width=2).eval()
    path = tmp_path / "ternary.pt"
    save_ternary_checkpoint(
        model,
        path,
        model_name="r18",
        dataset="mnist",
        width=2,
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert is_ternary_checkpoint(payload)
    assert payload["format"] == TERNARY_CHECKPOINT_FORMAT
    assert payload["model"] == "r18"

    restored = freeze_ternary_weights(build_resnet("r18", in_channels=1, width=2))
    restored.load_state_dict(payload["state_dict"])
    ternary_weights = [
        module.weight
        for module in restored.modules()
        if isinstance(module, (Bit.Conv2dInfer, Bit.LinearInfer))
    ]
    assert ternary_weights
    assert all(weight.dtype == torch.int8 for weight in ternary_weights)
    assert all(torch.all((weight >= -1) & (weight <= 1)) for weight in ternary_weights)
