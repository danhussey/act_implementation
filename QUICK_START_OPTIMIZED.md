# Quick Start: Optimized Data Collection

## TL;DR - Fastest Path to Results

### Option 1: Quick Test (1 hour)
```bash
# Collect 50 episodes with optimizations (30-60 min)
uv run python collect_demos_parallel.py \
  --episodes 50 \
  --workers 4 \
  --output data/quick_test.hdf5

# Train for 10 epochs to verify (5-10 min on CPU, 1-2 min on CUDA)
uv run python train.py \
  --data data/quick_test.hdf5 \
  --epochs 10 \
  --chunk-size 90 \
  --device cuda  # or 'mps' on Mac, 'cpu' otherwise
```

### Option 2: Paper-Spec Collection (3-7 hours on remote PC)
```bash
# Collect 100 episodes with ACT paper specs
uv run python collect_demos_parallel.py \
  --episodes 100 \
  --workers 8 \
  --camera-height 480 \
  --camera-width 640 \
  --output data/paper_spec_demos.hdf5

# Train with full optimizations
uv run python train.py \
  --data data/paper_spec_demos.hdf5 \
  --epochs 100 \
  --chunk-size 90 \
  --batch-size 64 \
  --device cuda \
  --use-amp \
  --compile
```

---

## Optimizations Enabled by Default

### Parallel Collection
- ✅ Multiprocessing enabled (4-8 workers)
- ✅ Isolated RoboSuite environments per worker
- ✅ Automatic result merging

### Fast Scripted Policy
- ✅ 2x base speed multiplier
- ✅ Adaptive speed (4x in free space, 0.5x near target)
- ✅ Concurrent gripper control
- ✅ Early success detection
- ✅ Adaptive tolerances (loose/tight)

### ACT Paper Specs
- ✅ 50Hz control frequency
- ✅ Chunk size 90
- ✅ 2 cameras for single-arm (agentview + wrist)
- 📝 480×640 resolution (specify with --camera-height/--camera-width)

---

## Command Options

### Parallel Collection
```bash
uv run python collect_demos_parallel.py \
  --episodes 100        # Total episodes to collect
  --workers 8           # Number of parallel workers (match CPU cores)
  --camera-height 480   # Image height (84 or 480)
  --camera-width 640    # Image width (84 or 640)
  --output data/demos.hdf5  # Output file
  --noise 0.005         # Action noise for diversity
  --keep-failed         # Keep failed episodes (default: filter them)
```

### Training
```bash
uv run python train.py \
  --data data/demos.hdf5     # Input dataset
  --chunk-size 90            # Action chunk size (ACT paper: 90)
  --epochs 100               # Training epochs
  --batch-size 64            # Batch size
  --device cuda              # Device: cuda/mps/cpu
  --use-amp                  # Mixed precision (CUDA only, 2-3x faster)
  --compile                  # torch.compile (PyTorch 2.0+, 1.5-2x faster)
  --output-dir checkpoints   # Checkpoint directory
  --save-freq 10             # Save checkpoint every N epochs
```

---

## Hardware Recommendations

### For Data Collection (CPU-bound)
- **Minimum:** 4 CPU cores, 8GB RAM → 4 workers
- **Recommended:** 8 CPU cores, 16GB RAM → 8 workers
- **Note:** More cores = proportionally faster collection

### For Training (GPU-bound)
- **Mac:** Use `--device mps` (Apple Silicon)
- **Linux/Windows with NVIDIA:** Use `--device cuda --use-amp --compile`
- **CPU only:** Use `--device cpu` (slower but works)

**GPU recommendations:**
- RTX 3060 12GB: Best value (~$300)
- RTX 4060 Ti 16GB: More VRAM (~$450)
- RTX 4070: Faster training (~$550)

---

## Expected Performance

### Collection Time (100 episodes)

| Configuration | Resolution | Workers | Time |
|--------------|-----------|---------|------|
| Baseline (serial) | 84×84 | 1 | 16 hours |
| **Optimized** | 84×84 | 4 | **1.5-2.5 hours** |
| **Optimized** | 84×84 | 8 | **0.8-1.3 hours** |
| Paper-spec (serial) | 480×640 | 1 | 40-50 hours |
| **Paper-spec (optimized)** | 480×640 | 8 | **3-7 hours** |

### Training Time (100 epochs)

| Device | Time/Epoch | Total (100 epochs) |
|--------|-----------|-------------------|
| Mac M1/M2 (MPS) | 2-3 min | 3-5 hours |
| CPU only | 8 min | 13 hours |
| RTX 3060 (no opt) | 2 min | 3.5 hours |
| **RTX 3060 (--use-amp --compile)** | **20-40 sec** | **1-2 hours** |

---

## Troubleshooting

### "Out of memory" during parallel collection
```bash
# Reduce number of workers
--workers 2
```

### "CUDA out of memory" during training
```bash
# Reduce batch size
--batch-size 32
```

### Parallel collection not faster
- Check CPU usage (`htop` or Task Manager)
- Ensure workers are actually running in parallel
- Try different worker counts (2, 4, 8)

### Episodes taking too long
- Verify optimized policy is being used (check for speed_multiplier=2.0)
- Check RoboSuite version (should be 1.5.1+)
- Monitor with `nvidia-smi` or `htop` to ensure no bottlenecks

---

## Benchmarking

To measure actual speedup on your hardware:

```bash
# Test with 5 episodes to estimate performance
uv run python scripts/benchmark_collection.py \
  --episodes 5 \
  --test-parallel \
  --workers 2 4 8
```

This will show actual speedup for your specific hardware.

---

## Next Steps

1. **Test pipeline:** Run quick test (50 episodes, 84×84)
2. **Validate training:** Verify loss decreases
3. **Scale to paper-spec:** Deploy to remote PC, collect 100 episodes at 480×640
4. **Full training:** 100 epochs with `--use-amp --compile` on CUDA

See `OPTIMIZATION_SUMMARY.md` for detailed technical explanation of all optimizations.
