# Remote CUDA Training Setup

Quick guide for running ACT training on a remote Linux PC with CUDA.

## 1. Transfer Repository

```bash
# From your Mac - push to git
git push origin main

# OR - direct sync via rsync
rsync -avz --exclude='.venv' --exclude='__pycache__' \
  ~/Code/act_implementation/ user@remote-pc:~/act_implementation/
```

## 2. Setup on Remote PC

```bash
# SSH into remote
ssh user@remote-pc
cd ~/act_implementation

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv venv
source .venv/bin/activate
uv sync

# Verify CUDA
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 3. Generate Mock Data (Recommended)

**Fast option** - generates 100 episodes in ~2 seconds:

```bash
uv run python scripts/generate_mock_data.py \
  --episodes 100 \
  --output data/mock_demos.hdf5
```

**Slow option** - real RoboSuite data (10-15 min/episode = 16-25 hours for 100):

```bash
# May need virtual display on headless Linux
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &

uv run python collect_demos.py \
  --env PickPlaceCan \
  --episodes 100 \
  --output data/real_demos.hdf5
```

## 4. Train with Optimizations

**Quick test** (2 epochs):
```bash
uv run python train.py \
  --data data/mock_demos.hdf5 \
  --epochs 2 \
  --batch-size 64 \
  --device cuda \
  --use-amp \
  --compile
```

**Full training** (100 epochs, ~1-2 hours with optimizations):
```bash
uv run python train.py \
  --data data/mock_demos.hdf5 \
  --epochs 100 \
  --batch-size 64 \
  --device cuda \
  --use-amp \
  --compile \
  --output-dir checkpoints/cuda_training \
  --save-freq 10
```

### Optimization Flags:
- `--use-amp`: Mixed precision (2-3x faster, CUDA only)
- `--compile`: torch.compile (1.5-2x faster, PyTorch 2.0+)
- Combined: **3-5x speedup**

## 5. Monitor Training

```bash
# In another terminal
uv run tensorboard --logdir logs --bind_all
# Open http://remote-pc:6006 in browser
```

Or monitor GPU:
```bash
watch -n 1 nvidia-smi
```

## 6. Transfer Models Back

```bash
# From your Mac
rsync -avz user@remote-pc:~/act_implementation/checkpoints/ \
  ~/Code/act_implementation/checkpoints/

# Or specific checkpoint
scp user@remote-pc:~/act_implementation/checkpoints/best_model.pt \
  ~/Code/act_implementation/checkpoints/
```

## Expected Performance

**Without optimizations:**
- ~2 minutes/epoch
- 100 epochs = ~3.5 hours

**With `--use-amp --compile`:**
- ~20-40 seconds/epoch
- 100 epochs = ~1-2 hours

**vs Mac CPU:**
- ~8 minutes/epoch
- 100 epochs = ~13 hours

## Troubleshooting

**CUDA Out of Memory:**
```bash
# Reduce batch size
--batch-size 32
```

**torch.compile errors:**
```bash
# Remove --compile flag or update PyTorch
uv pip install --upgrade torch
```

**RoboSuite display issues:**
```bash
# Install and start Xvfb
sudo apt-get install xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```
