# ACT in One File

Small PyTorch implementation of Action Chunking Transformer-style behavior
cloning for HDF5 robot demonstrations.

The repo is intentionally minimal:

- `act.py` contains the dataset loader, model, mock-data writer, and training CLI.
- `tests/test_act.py` smoke-tests the script.
- Generated data, checkpoints, logs, and runs are ignored.

## Install

```bash
uv sync --group dev
```

## Smoke Run

```bash
uv run python act.py mock --out data/mock.hdf5 --episodes 8 --steps 32
uv run python act.py train --data data/mock.hdf5 --out runs/smoke --epochs 3 --batch-size 8
```

Outputs:

- `runs/smoke/best.pt`
- `runs/smoke/metrics.json`

## Data Format

`act.py train` expects an HDF5 file like:

```text
/episode_0/states              (T, state_dim)
/episode_0/actions             (T, action_dim)
/episode_0/images_agentview    (T, 3, H, W)
/episode_0/images_wrist        (T, 3, H, W)
attrs["num_episodes"] = N
```

Any dataset named `images_*` is treated as a camera.
