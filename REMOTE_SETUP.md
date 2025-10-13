# Remote CUDA Setup Guide

Instructions for running ACT training on a remote Linux PC with CUDA.

## Step 1: Transfer Repository to Remote

```bash
# On your local Mac - push to remote repo (if using git remote)
git push origin main

# OR - sync directly via rsync
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  ~/Code/act_implementation/ user@remote-pc:~/act_implementation/
```

## Step 2: Setup on Remote PC

```bash
# SSH into remote PC
ssh user@remote-pc

# Navigate to project
cd ~/act_implementation

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## Step 3: Generate Mock Data (Quick Test)

```bash
# Generate mock data for quick testing
uv run python scripts/generate_mock_data.py --episodes 100 --output data/mock_demos.hdf5
```

## Step 4: Run Training with CUDA

```bash
# Quick test (2 epochs)
uv run python train.py \
  --data data/mock_demos.hdf5 \
  --epochs 2 \
  --batch-size 64 \
  --device cuda \
  --output-dir checkpoints/test_cuda

# Full training run
uv run python train.py \
  --data data/mock_demos.hdf5 \
  --epochs 100 \
  --batch-size 64 \
  --device cuda \
  --output-dir checkpoints/mock_training \
  --lr 1e-4 \
  --save-freq 10

# Monitor training with TensorBoard (in another terminal)
tensorboard --logdir logs --bind_all
```

## Step 5: Collect Real RoboSuite Data (Optional)

If you want real demonstrations instead of mock data:

```bash
# Install RoboSuite dependencies (may require display server)
uv add robosuite

# Collect demonstrations
uv run python collect_demos.py \
  --env PickPlaceCan \
  --episodes 100 \
  --output data/real_demos.hdf5 \
  --noise 0.005

# Train on real data
uv run python train.py \
  --data data/real_demos.hdf5 \
  --epochs 100 \
  --batch-size 64 \
  --device cuda \
  --output-dir checkpoints/real_training
```

## Step 6: Transfer Trained Models Back to Mac

```bash
# From your Mac
rsync -avz user@remote-pc:~/act_implementation/checkpoints/ \
  ~/Code/act_implementation/checkpoints/

# Or download specific checkpoint
scp user@remote-pc:~/act_implementation/checkpoints/best_model.pt \
  ~/Code/act_implementation/checkpoints/
```

## Performance Expectations

**CUDA GPU Training:**
- ~1-2 seconds per batch (64 samples)
- ~1-2 minutes per epoch (with 100 episodes, batch size 64)
- Full training (100 epochs): ~2-3 hours

**vs CPU (Mac):**
- ~8-10 minutes per epoch
- Full training would take ~13-16 hours

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
uv run python train.py --data data/mock_demos.hdf5 --batch-size 32 --device cuda
```

### RoboSuite Display Issues
RoboSuite may require a display server on headless Linux:
```bash
# Use Xvfb (virtual display)
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
uv run python collect_demos.py --env PickPlaceCan --episodes 10
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Next Steps After Training

1. **Evaluate trained model:**
   ```bash
   uv run python eval.py \
     --checkpoint checkpoints/best_model.pt \
     --data data/mock_demos.hdf5 \
     --episodes 50 \
     --device cuda
   ```

2. **View training curves:**
   - Open TensorBoard in browser: `http://remote-pc:6006`
   - Or sync logs to local: `rsync -avz user@remote-pc:~/act_implementation/logs/ logs/`
