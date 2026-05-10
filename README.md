# ACT in One File

Small PyTorch implementation of Action Chunking Transformer-style behavior
cloning for real robomimic low-dimensional demonstrations.

The repo is intentionally minimal:

- `act.py` contains the downloader, robomimic HDF5 loader, model, training CLI, and rollout-to-mp4 path.
- `ACT_walkthrough.ipynb` explains the flow step by step with shape prints.
- `tests/test_act.py` smoke-tests the script.
- Generated data, checkpoints, logs, and runs are ignored.

## Install

```bash
uv sync --group dev
```

## Download And Train

```bash
uv run python act.py download --dataset lift-ph
uv run python act.py train --data data/lift_ph_low_dim.hdf5 --out runs/lift --epochs 3 --batch-size 64
```

Outputs:

- `runs/lift/best.pt`
- `runs/lift/metrics.json`

## Rollout To MP4

```bash
uv run python act.py rollout \
  --checkpoint runs/lift/best.pt \
  --data data/lift_ph_low_dim.hdf5 \
  --out runs/lift/rollout.mp4 \
  --device mps
```

This replays the trained policy in robosuite from one demonstration's initial
simulator state and writes an mp4 through the local `ffmpeg` binary.

## Evaluate Rollouts

```bash
uv run python act.py evaluate \
  --checkpoint runs/lift/best.pt \
  --data data/lift_ph_low_dim.hdf5 \
  --out-dir runs/lift_eval \
  --episodes 20 \
  --videos 3 \
  --device mps
```

Outputs:

- `runs/lift_eval/eval_metrics.json`
- `runs/lift_eval/videos/rollout_*.mp4`

`lift-ph` is the default because it is the smallest robomimic v1.5 low-dimensional file. `can-ph` is also available:

```bash
uv run python act.py download --dataset can-ph
```

Training reads robomimic HDF5 files directly from `/data/demo_*/obs/*` and `/data/demo_*/actions`. It concatenates all low-dimensional observation arrays in each `obs` group into the policy state.

For a more guided explanation, open `ACT_walkthrough.ipynb`.
