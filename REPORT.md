# ACT Implementation Report

## Summary

This repository is a compact, readable Action Chunking Transformer implementation for learned robotic manipulation policies. It is designed as a research-engineering artifact: small enough to inspect, but complete enough to train on robomimic demonstrations, evaluate closed-loop simulator rollouts, save videos, and compare supervised losses against actual task success.

## Setup

The implementation is centered on `act.py` and uses `uv` for the local environment. The main dependencies are PyTorch, robomimic/robosuite data and simulation tooling, NumPy, h5py, and pytest for smoke tests.

```bash
uv sync --group dev
uv run python act.py download --dataset lift-ph
uv run python act.py train --data data/lift_ph_low_dim.hdf5 --out runs/lift --epochs 3 --batch-size 64 --device mps
uv run python act.py evaluate --checkpoint runs/lift/best.pt --data data/lift_ph_low_dim.hdf5 --out-dir runs/lift_eval --episodes 20 --videos 3 --device mps
```

Use `--device cpu`, `--device mps`, or `--device cuda` depending on local hardware.

## Model Architecture

The model follows the ACT idea of predicting a short chunk of future actions from the current observation rather than predicting only the next action. The implementation includes:

- a low-dimensional observation path for robot state plus privileged simulator state,
- an image-observation path using `agentview_image` plus proprioception,
- a scratch CNN encoder for image mode,
- an optional frozen pretrained ResNet-18 image encoder,
- a CVAE-style latent path for multimodal action chunks,
- checkpointing for both best validation loss and final training state.

The low-dimensional mode is intentionally privileged: it gets simulator object state. The image mode removes that shortcut and tests whether the policy can infer useful object information from pixels.

## Data

Experiments use public robomimic demonstrations:

| Dataset | Observation mode | Role |
| --- | --- | --- |
| `lift-ph` | Low-dimensional state | Quick privileged-state baseline |
| `can-ph` | Low-dimensional state | Harder manipulation baseline |
| `lift-ph-image` | Camera image plus proprioception | First high-dimensional comparison |

Generated datasets, rendered image observations, checkpoints, metrics, and videos are intentionally kept out of git under `data/` and `runs/`.

## Experiments

The core experiment loop is:

1. Download or render demonstration data.
2. Train ACT on expert observation/action chunks.
3. Save `best.pt` by supervised validation loss and `last.pt` at the end of training.
4. Roll out the policy in robosuite from demonstration initial states.
5. Record success counts, rewards, rollout logs, curves, and MP4/GIF clips.

This matters because imitation-learning validation loss and closed-loop task success are related but not interchangeable.

## Results

| Task | Run | Checkpoint | Horizon | Success | Readout |
| --- | --- | --- | ---: | ---: | --- |
| `lift-ph` | 3 epochs | `best.pt` | 100 | 14/20 | Minimal low-dim training already learns useful behavior. |
| `lift-ph` | 20 epochs | `best.pt` | 100 | 10/20 | Better validation loss did not mean better rollout behavior. |
| `lift-ph` | low-dim 20 minutes | `best.pt` | 100 | 18/20 | Strongest privileged-state Lift run so far. |
| `lift-ph-image` | scratch CNN, 20 epochs | `best.pt` | 100 | 10/20 | First high-dimensional baseline from pixels plus proprioception. |
| `lift-ph-image` | scratch CNN, +40 minutes | `last.pt` | 100 | 17/20 | Best vision run so far. |
| `can-ph` | 1 epoch smoke | `best.pt` | 200 | 1/20 | Barely trained baseline. |
| `can-ph` | 6 hr validation-best | `best.pt` | 200 | 7/20 | Lowest supervised validation loss was not the best actor. |
| `can-ph` | 6 hr final | `last.pt` | 200 | 18/20 | Strongest can run so far. |

## Key Finding

Supervised validation loss was a poor selector of closed-loop policy quality in the most interesting runs. On `can-ph`, the validation-best checkpoint reached 7/20 successes, while the final checkpoint reached 18/20. On vision Lift, the final checkpoint reached 17/20, while the validation-best checkpoint reached 13/20.

The practical lesson is that validation loss is useful for debugging learning, but closed-loop rollout success is the metric that answers whether the policy can actually complete the task once its own actions change the next state.

## Failure Modes

- Low-dimensional policies can look strong because they receive privileged simulator state.
- Image policies are more brittle and currently depend on rendered training observations.
- Validation-best checkpoints can underperform final checkpoints during rollout.
- A policy can imitate local action chunks yet drift into states where its next predictions are poor.
- Short-horizon success may hide failure modes that appear with longer rollout horizons or harder tasks.

## Next Experiments

- Add scheduled rollout probes during longer training runs and select checkpoints by closed-loop success.
- Compare scratch CNN and frozen ResNet encoders on the same rendered image dataset.
- Run more seeds for the `can-ph` final-vs-validation-best result.
- Add stronger video/result summaries for failure categories.
- Try a non-privileged state estimator or image-pretrained encoder before claiming vision robustness.

## Reproducibility Notes

The repository keeps generated artifacts out of git, but each run writes machine-readable logs such as `metrics.json`, `history.jsonl`, `rollout_history.jsonl`, rollout metrics, curves, and videos. Tests are intentionally lightweight and focus on keeping the one-file implementation executable as the experiment surface changes.
