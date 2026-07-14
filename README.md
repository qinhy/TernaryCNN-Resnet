# TernaryCNN-ResNet

Experiments, tests, and command-line tools for ResNet models built with
[TernaryCNN](https://github.com/qinhy/TernaryCNN).

The project supports ResNet-18, ResNet-50, and ResNet-101 on MNIST, CIFAR-10,
and CIFAR-100. It selects the input channels and classifier size automatically.
The small-image stem and configurable first-stage width make quick local
experiments practical.

## Setup

This repository is configured to use an adjacent editable TernaryCNN checkout:

```text
parent/
├── TernaryCNN/
└── TernaryCNN-Resnet/
```

Install the project and development tools:

```bash
uv sync --group dev
```

For a standalone install without the adjacent checkout, remove `[tool.uv.sources]`
from `pyproject.toml`; uv will install TernaryCNN from its Git repository.

## Train

Run a no-download smoke test with generated MNIST-shaped images:

```bash
uv run train-resnet --smoke-test
uv run train-resnet --dataset cifar10 --smoke-test
uv run train-resnet --dataset cifar100 --smoke-test
```

Train the default compact ResNet-18 on a subset of MNIST:

```bash
uv run train-resnet
```

Train on the complete dataset and save a checkpoint:

```bash
uv run train-resnet --train-samples 0 --test-samples 0 --epochs 5 \
  --save checkpoints/mnist-r18.pt \
  --save-ternary checkpoints/mnist-r18-ternary.pt
```

`--save` writes the trainable floating-point master weights used for QAT and
future training. `--save-ternary` writes a frozen inference checkpoint whose
quantized convolution and linear weights are stored as `int8` values in
`{-1, 0, 1}`, together with their per-output-channel scales. You can use either
option or both in the same run.

Select a larger architecture with `--model r50` or `--model r101`. Run
`uv run train-resnet --help` for all options.

Train each real dataset by selecting its name:

```bash
uv run train-resnet --dataset mnist
uv run train-resnet --dataset cifar10
uv run train-resnet --dataset cifar100
```

## Evaluate

```bash
uv run evaluate-resnet checkpoints/mnist-r18.pt
uv run evaluate-resnet checkpoints/mnist-r18-ternary.pt
```

Frozen ternary checkpoints include the dataset, architecture, and width, so
evaluation selects them automatically. Legacy trainable state dictionaries use
the defaults unless matching `--dataset`, `--model`, and `--width` options are
provided.

## Test and lint

```bash
uv run pytest
uv run ruff check .
```

## Practical training commands

The commands below use every training and test image (`--train-samples 0` and
`--test-samples 0`) and save separate checkpoints. `--device auto` selects CUDA,
MPS, or CPU in that order. The deeper CIFAR runs are intended for a GPU.

### MNIST

```bash
uv run train-resnet --dataset mnist --model r18 --width 32 --epochs 10 \
  --train-samples 0 --test-samples 0 --device auto \
  --save checkpoints/mnist-r18.pt \
  --save-ternary checkpoints/mnist-r18-ternary.pt

uv run train-resnet --dataset mnist --model r50 --width 32 --epochs 10 \
  --train-samples 0 --test-samples 0 --device auto \
  --save checkpoints/mnist-r50.pt \
  --save-ternary checkpoints/mnist-r50-ternary.pt

uv run train-resnet --dataset mnist --model r101 --width 32 --epochs 10 \
  --train-samples 0 --test-samples 0 --device auto \
  --save checkpoints/mnist-r101.pt \
  --save-ternary checkpoints/mnist-r101-ternary.pt
```

### CIFAR-10

```bash
uv run train-resnet --dataset cifar10 --model r18 --width 64 --epochs 200 \
  --train-samples 0 --test-samples 0 --device auto \
  --save checkpoints/cifar10-r18.pt \
  --save-ternary checkpoints/cifar10-r18-ternary.pt

uv run train-resnet --dataset cifar10 --model r50 --width 64 --epochs 200 \
  --train-samples 0 --test-samples 0 --device auto \
  --save checkpoints/cifar10-r50.pt \
  --save-ternary checkpoints/cifar10-r50-ternary.pt

uv run train-resnet --dataset cifar10 --model r101 --width 64 --epochs 200 \
  --train-samples 0 --test-samples 0 --device auto \
  --save checkpoints/cifar10-r101.pt \
  --save-ternary checkpoints/cifar10-r101-ternary.pt
```

### CIFAR-100

```bash
uv run train-resnet --dataset cifar100 --model r18 --width 64 --epochs 200 \
  --train-samples 0 --test-samples 0 --device auto \
  --save checkpoints/cifar100-r18.pt \
  --save-ternary checkpoints/cifar100-r18-ternary.pt

uv run train-resnet --dataset cifar100 --model r50 --width 64 --epochs 200 \
  --train-samples 0 --test-samples 0 --device auto \
  --save checkpoints/cifar100-r50.pt \
  --save-ternary checkpoints/cifar100-r50-ternary.pt

uv run train-resnet --dataset cifar100 --model r101 --width 64 --epochs 200 \
  --train-samples 0 --test-samples 0 --device auto \
  --save checkpoints/cifar100-r101.pt \
  --save-ternary checkpoints/cifar100-r101-ternary.pt
```
