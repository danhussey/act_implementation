# ACT Implementation

In this repo I'm reimplementing the Action Chunking Transformer paper from scratch, to perform a Pick Place task.

## Project Structure

```
act_implementation/
├── models/          # Vision encoder, ACT transformer, CVAE
├── data/            # Dataset loading and preprocessing
├── envs/            # RoboSuite environment wrapper and scripted policy
├── utils/           # Utility functions
collect_demos.py     # Data collection script
train.py             # Training script
eval.py              # Evaluation script with temporal ensembling
```

## Setup

Install dependencies using UV:
```bash
uv sync
```

## Usage

### 1. Collect Demonstrations

Generate synthetic demonstrations using the scripted policy:
```bash
uv run python collect_demos.py --env PickPlaceCan --episodes 100 --output data/demos.hdf5
```

### 2. Train ACT Model

Train the model on collected demonstrations:
```bash
uv run python train.py --data data/demos.hdf5 --epochs 100 --device mps
```

For CUDA (on remote PC):
```bash
uv run python train.py --data data/demos.hdf5 --epochs 100 --device cuda
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs
```

### 3. Evaluate Policy

Evaluate the trained policy with temporal ensembling:
```bash
uv run python eval.py --checkpoint checkpoints/best_model.pt --data data/demos.hdf5 --episodes 50
```

## Key Features

- **Action Chunking**: Predicts sequences of future actions for temporal consistency
- **CVAE**: Conditional VAE for handling multimodal action distributions
- **Temporal Ensembling**: Averages overlapping action predictions during inference
- **Cross-Platform**: Supports MPS (macOS), CUDA (Linux), and CPU
- **Vision Encoder**: ResNet-based encoder with spatial softmax option
- **Scripted Data Collection**: Automated demonstration generation

## Model Architecture

- **Vision Encoder**: ResNet18 backbone for processing multi-camera RGB observations
- **Transformer**: 4-layer encoder-decoder with 8 attention heads
- **CVAE**: 32-dimensional latent space for action distribution modeling
- **Chunk Size**: 10 timesteps (configurable)

## Training Details

- **Loss**: MSE reconstruction + KL divergence (weight=10.0)
- **Optimizer**: AdamW with cosine annealing
- **Learning Rate**: 1e-4
- **Batch Size**: 32
- **Normalization**: State and action normalization using dataset statistics
