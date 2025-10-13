# Data Collection Optimization Summary

This document summarizes all optimizations made to speed up real RoboSuite data collection while maintaining ACT paper specifications.

## Completed Optimizations

### 1. Parallel Collection (4-8x speedup) ✅
**File:** `collect_demos_parallel.py`

**Changes:**
- Implemented multiprocessing with `Pool` to run 4-8 RoboSuite environments simultaneously
- Each worker collects episodes independently and saves to temporary HDF5 files
- Results are merged into final dataset
- Uses `spawn` method to avoid memory leaks with RoboSuite

**Usage:**
```bash
# Collect 100 episodes with 8 workers
uv run python collect_demos_parallel.py \
  --episodes 100 \
  --workers 8 \
  --output data/demos.hdf5 \
  --camera-height 480 \
  --camera-width 640
```

**Expected speedup:** With 8 cores, 8-hour collection → 1-hour collection

---

### 2. Optimized Scripted Policy (2-3x speedup) ✅
**File:** `act_implementation/envs/scripted_policy.py`

**Changes implemented:**

#### a) Adaptive Speed Control
- Base speed multiplier: 2.0x (configurable)
- Distance-based scaling:
  - **>0.15m:** 2x speed (fast in free space)
  - **0.08-0.15m:** 1.5x speed (medium)
  - **0.03-0.08m:** 1.0x speed (normal near target)
  - **<0.03m:** 0.5x speed (slow for precision)

#### b) Concurrent Actions
- **APPROACH_OBJECT:** Starts closing gripper when within 0.15m of object (gradual closure)
- **PLACE:** Starts opening gripper when within 0.05m of placement height

#### c) Adaptive Tolerances
- **Loose tolerance (0.08m):** Used for free space movement (APPROACH, MOVE_TO_TARGET, RETREAT)
- **Default tolerance (0.05m):** Used for lift
- **Tight tolerance (0.03m):** Used for precision placement

#### d) Early Success Detection
- Checks if object is already at target (within 0.08m x-y distance)
- Requires stability for 10 steps to avoid false positives
- Terminates episode early if task is complete

**Configuration:**
```python
policy = ScriptedPickPlacePolicy(
    speed_multiplier=2.0,  # 2x faster base speed
    position_tolerance=0.05,  # Default
    position_tolerance_loose=0.08,  # Free space
    position_tolerance_tight=0.03,  # Precision
    grasp_threshold=0.06,  # Slightly larger for faster grasping
)
```

**Expected speedup:** 2-3x faster episode completion

---

### 3. ACT Paper Specifications ✅

#### Control Frequency
**File:** `act_implementation/envs/robosuite_wrapper.py`

- Added `control_freq` parameter (default: 50Hz)
- Matches ACT paper specification

**Usage:**
```python
env = RoboSuiteWrapper(
    env_name="PickPlaceCan",
    control_freq=50,  # ACT paper uses 50Hz
)
```

#### Chunk Size
**File:** `train.py`

- Updated default `chunk_size` from 10 to 90
- Matches ACT paper specification

**Usage:**
```bash
uv run python train.py \
  --data data/demos.hdf5 \
  --chunk-size 90  # ACT paper default
```

#### Image Resolution
- ACT paper uses **480×640** resolution
- Current scripts still default to 84×84 for testing
- Update camera parameters for paper-spec collection:

```bash
uv run python collect_demos_parallel.py \
  --camera-height 480 \
  --camera-width 640  # Match ACT paper
```

---

### 4. Benchmark Script ✅
**File:** `scripts/benchmark_collection.py`

Measures collection speed with different configurations.

**Usage:**
```bash
# Test parallel collection with different worker counts
uv run python scripts/benchmark_collection.py \
  --episodes 5 \
  --test-parallel \
  --workers 2 4 8

# Test different resolutions
uv run python scripts/benchmark_collection.py \
  --episodes 5 \
  --test-resolution \
  --resolutions 84x84 256x256 480x640
```

---

## Performance Projections

### Current Setup (84×84, Optimized)
| Configuration | Time/Episode | 100 Episodes |
|--------------|--------------|--------------|
| **Serial (baseline)** | 10 min | 16-17 hours |
| **Serial + fast policy** | 4-5 min | 6-8 hours |
| **Parallel (4 workers) + fast policy** | 1-1.5 min | 1.5-2.5 hours |
| **Parallel (8 workers) + fast policy** | 0.5-0.8 min | 0.8-1.3 hours |

### Paper Spec (480×640, 50Hz)
| Configuration | Time/Episode | 100 Episodes |
|--------------|--------------|--------------|
| **Serial (no optimization)** | 25-30 min | 40-50 hours |
| **Serial + fast policy** | 12-15 min | 20-25 hours |
| **Parallel (4 workers) + fast policy** | 4-6 min | 6-10 hours |
| **Parallel (8 workers) + fast policy** | 2-4 min | 3-7 hours |

---

## Recommended Collection Workflow

### Phase 1: Test Pipeline (Fast)
Use smaller resolution to verify everything works:

```bash
# Generate 50 episodes quickly for testing
uv run python collect_demos_parallel.py \
  --episodes 50 \
  --workers 4 \
  --camera-height 84 \
  --camera-width 84 \
  --output data/test_demos.hdf5

# Train to verify pipeline
uv run python train.py \
  --data data/test_demos.hdf5 \
  --epochs 10 \
  --chunk-size 90
```

**Expected time:** 30-60 minutes for collection

### Phase 2: Paper-Spec Collection (Remote PC)
Deploy to CUDA PC and collect full dataset:

```bash
# On remote PC with CUDA
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

**Expected time:** 3-7 hours for collection, 1-2 hours for training

---

## Pending Optimizations

### Infrastructure Improvements (optional, 1.2-1.5x speedup)
These could provide additional minor speedups but are not essential:

1. **Async HDF5 Writing**
   - Use queue-based writer thread
   - Don't block simulation for disk I/O

2. **JPEG Compression**
   - Compress images during collection (quality=90)
   - Reduces file size 10-20x

3. **Pre-allocated Arrays**
   - Pre-allocate episode arrays instead of appending
   - Avoid repeated numpy concatenation

These optimizations have diminishing returns compared to parallelization + fast policy.

---

## Key Takeaways

1. **Parallelization is essential** - Provides 4-8x speedup, no quality loss
2. **Optimized policy** - 2-3x speedup through smart movement and tolerances
3. **Combined speedup: 8-24x** - Makes paper-spec collection viable (40-50 hours → 3-7 hours)
4. **ACT specs matter** - 480×640, 50Hz, chunk_size=90 for paper-faithful implementation
5. **Test first** - Use 84×84 to validate pipeline, then scale to 480×640

---

## Files Modified

**New files:**
- `collect_demos_parallel.py` - Parallel collection with multiprocessing
- `scripts/benchmark_collection.py` - Benchmark different configurations
- `OPTIMIZATION_SUMMARY.md` - This document

**Modified files:**
- `act_implementation/envs/scripted_policy.py` - Speed optimizations
- `act_implementation/envs/robosuite_wrapper.py` - Added control_freq parameter
- `train.py` - Updated default chunk_size to 90

---

## Questions?

If you encounter issues:
1. Check that RoboSuite is properly installed
2. Verify sufficient RAM for parallel workers (each ~500MB)
3. Test with 2 workers first before scaling to 8
4. Use benchmark script to measure actual speedup on your hardware
