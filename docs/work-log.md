# Work Log

Concise project context notes. Preserve surprises, metrics, decisions, useful commands,
artifacts, and next questions by day.

## 2026-05-10

- Repo is currently a compact one-file ACT demo around `act.py`: download robomimic low-dim data, train, rollout, and evaluate.
- Dataset surprise: `lift-ph` low-dim is not vision. Each timestep is 53 scalar features from robot/object state, including a 10D `object` vector, end-effector pose/quaternion, gripper pos/vel, joint pos/vel, and joint sin/cos.
- Training run: `runs/lift_mps_run_20260510`, 20 demos, 3 epochs on MPS, best val loss `0.6819`.
- Single rollout: `runs/lift_mps_run_20260510/rollout.mp4`, success `true`, reward `1.0`, 54 steps.
- Batch evaluation: 20 rollout attempts from demo initial states, 14 successes, success rate `70%`, avg reward `0.7`, avg length `64.45` steps.
- Eval artifacts: `runs/lift_mps_run_20260510/eval_20/eval_metrics.json` and three sample videos in `runs/lift_mps_run_20260510/eval_20/videos/`.
- Open question: current success rate is from privileged low-dim simulator state, not camera observations. Vision ACT would require image observations plus a visual encoder path.
- Latent sweep command added: `act.py latent-sweep` runs `z=0` plus sampled fixed latents from the same start state and saves comparison videos.
- Latent sweep result on `demo_0`: 9 latent choices, 4 successes, success rate `44.4%`; artifacts in `runs/lift_mps_run_20260510/z_standard_demo0/`.
- Amplified latent sweep: same sweep with `--scale 2.0` gave 3/9 successes, success rate `33.3%`; artifacts in `runs/lift_mps_run_20260510/z_amplified_2x_demo0/`.
- Latent sweep surprise: first predicted actions were very similar across latents, but rollout outcomes diverged. The latent effect appears to compound over the trajectory rather than showing as an obvious first-step mode switch.

## 2026-05-11

- Longer training run: `uv run python act.py train --data data/lift_ph_low_dim.hdf5 --out runs/lift_mps_20ep_20260511 --epochs 20 --device mps`.
- Best validation loss improved from `0.6819` at 3 epochs to `0.4954` at 20 epochs, with the best checkpoint saved before the final epoch.
- Deterministic `z=0` eval from demo initial states regressed from 14/20 successes (`70%`) to 10/20 successes (`50%`), despite the better validation loss.
- Per-demo surprise: the longer model solved `demo_10` and `demo_13`, but lost `demo_0`, `demo_7`, `demo_9`, `demo_16`, `demo_17`, and `demo_18`.
- Artifacts: checkpoint and metrics in `runs/lift_mps_20ep_20260511/`; z=0 eval metrics in `runs/lift_mps_20ep_20260511/eval_20_z0/eval_metrics.json`; five eval MP4s in `runs/lift_mps_20ep_20260511/eval_20_z0/videos/`.
- Selected comparison videos for `demo_10`: `runs/lift_mps_20ep_20260511/compare_z0/baseline_3ep_demo_10.mp4` fails after 100 steps; `runs/lift_mps_20ep_20260511/compare_z0/long_20ep_demo_10.mp4` succeeds in 62 steps.
- Training command now supports `--max-minutes`, writes per-epoch `history.jsonl`, and saves both `best.pt` and `last.pt`.
- 20-minute run: `uv run python act.py train --data data/lift_ph_low_dim.hdf5 --out runs/lift_mps_20min_20260511 --epochs 10000 --max-minutes 20 --device mps`.
- 20-minute metrics: 256 epochs in `1203.7s`; best epoch `256`; best val loss `0.3687`; train loss `0.0453`.
- 20-minute `z=0` eval: 18/20 successes (`90%`), avg reward `0.9`, avg length `58.2` steps. Remaining failed starts: `demo_3`, `demo_15`.
- Learning: the 20-epoch regression was not evidence that more training generally hurts. It was probably an undertrained/noisy checkpoint. The broader warning still holds: chunk MSE is a weak proxy for closed-loop success, so rollouts remain the real metric.
- Artifacts: `runs/lift_mps_20min_20260511/metrics.json`, `runs/lift_mps_20min_20260511/history.jsonl`, `runs/lift_mps_20min_20260511/best.pt`, and five z=0 videos in `runs/lift_mps_20min_20260511/eval_20_z0/videos/`.
- Added `act.py plot-history` to turn `history.jsonl` into a no-dependency `loss_curve.svg` plus `history_summary.json`; generated the lift curve at `runs/lift_mps_20min_20260511/loss_curve.svg`.
- Downloaded `can-ph` low-dim dataset: 200 demos, avg length `116.0`, state dim `57`, action dim `7`; file at `data/can_ph_low_dim.hdf5`.
- Launched unattended six-hour can run in tmux session `can_mps_6h_20260511`: all 200 demos, batch size `64`, MPS, `--max-minutes 360`. It will train, plot history, then run 20 deterministic z=0 rollouts with five videos.
- Can run artifacts are under `runs/can_mps_6h_20260511/`; live log is `runs/can_mps_6h_20260511/train_and_eval.log`.

## 2026-05-12

- Six-hour `can-ph` run completed: 794 epochs in `21601.5s`, best val loss `0.1490` at epoch 18, final train loss `0.0355`, final val loss `0.1982`.
- First automatic eval used the old 100-step cap and got 0/20 successes. This was misleading for `can-ph`; demos average `116` steps and the task often needs more horizon.
- Re-eval with `--max-steps 200`: `best.pt` got 7/20 successes (`35%`), avg length `182.1`; `last.pt` got 18/20 successes (`90%`), avg length `116.9`.
- Main surprise: validation-best was not rollout-best. The final, more-overfit-looking checkpoint was much better closed-loop than the validation-best checkpoint.
- Remaining final-checkpoint failures in the 20-start eval: `demo_0`, `demo_11`.
- Artifacts: loss curve `runs/can_mps_6h_20260511/loss_curve.svg`; final-checkpoint eval `runs/can_mps_6h_20260511/eval_20_z0_h200_last/eval_metrics.json`; selected final videos in `runs/can_mps_6h_20260511/last_z0_selected/`.
