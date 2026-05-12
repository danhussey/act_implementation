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
- README graphics iteration: hand-authored SVGs clipped text in GitHub rendering, so they were replaced with Mermaid diagrams. Mermaid keeps the visuals deterministic and reviewable without manual coordinate tuning.
- Can smoke baseline eval: 1-epoch checkpoint got 1/20 successes (`5%`) with `--max-steps 200`; metrics at `runs/can_mps_smoke_20260511/eval_20_z0_h200/eval_metrics.json`.
- README now includes compact rollout GIFs generated from saved MP4s: lift 3-epoch failure vs 20-minute success on `demo_10`, and can 1-epoch failure vs 6-hour final success on `demo_1`.
- Started vision ACT path on branch `codex/vision-encoder`: image-mode training uses `agentview_image` plus robot proprioception and excludes the privileged `object` state vector.
- Added comparison knobs for `scratch_cnn` versus optional pretrained frozen `resnet18`; next useful metric is rollout success against the existing low-dim baseline, not just validation loss.
- Downloaded `lift-ph-image` raw demo file (`data/lift_ph_image.hdf5`, 200 demos, 35 MB). Surprise: it stores raw simulator `states` and `actions`, not pre-rendered `obs`; added `act.py render-images` to create compact image-observation HDF5 files before vision training.
- Rendered 20 Lift image demos at 128x128 from simulator states: `data/lift_ph_agentview_20demos.hdf5`, 1042 samples, about 17s after changing render reset to once per demo instead of once per timestep.
- Scratch-CNN 20-epoch smoke: `runs/lift_vision_scratch_20demos_20260512`, best val loss `0.4313`, z=0 rollout eval 10/20 successes (`50%`), matching the old low-dim 20-epoch count but on different starts.
- Scratch-CNN 20-minute run: `runs/lift_vision_scratch_20min_20260512`, 343 epochs in `1200.6s`, best val loss `0.3338` at epoch 80, final train loss `0.0649`, final val loss `0.4277`.
- Scratch-CNN 20-minute rollout eval: validation-best `best.pt` got 5/20 successes (`25%`), while `last.pt` got 9/20 (`45%`). This repeats the can-task lesson: validation loss can pick a worse closed-loop policy.
- Baseline comparison: low-dim Lift 20-minute checkpoint remains much stronger at 18/20 successes (`90%`). The scratch-CNN is learning from pixels but is not yet competitive with privileged low-dim state.
- Vision artifacts: loss curve at `runs/lift_vision_scratch_20min_20260512/loss_curve.svg`; best-checkpoint videos in `runs/lift_vision_scratch_20min_20260512/eval_20_z0/videos/`.
- Next question: compare against frozen pretrained ResNet-18 on the same rendered image file, then consider more demos or lighter augmentation before scaling the scratch CNN.
- Added rollout-aware training: `act.py train` now supports `--resume`, `--eval-before-train`, `--eval-every-epochs`, and `--eval-episodes`; probes write `rollout_history.jsonl`, and `plot-history` writes `rollout_curve.svg` when probes exist.
- Continued scratch-CNN Lift from `runs/lift_vision_scratch_20min_20260512/last.pt` for another 20 minutes with 10-rollout probes every 50 epochs: `runs/lift_vision_scratch_continue_rollout_20260512`.
- In-loop rollout probes improved from 4/10 at resume to 7/10 at epochs 200 and 300. Validation loss did not tell the same story: best val was `0.4227` at epoch 270, final val was worse at `0.4581`.
- Full 20-start eval after continuation: `best.pt` got 12/20 successes (`60%`), while `last.pt` got 14/20 (`70%`) with five MP4s in `runs/lift_vision_scratch_continue_rollout_20260512/eval_20_z0_last/videos/`.
- Learning: rollout probes are not overkill for this project. They add wall-clock cost, but they directly measure the closed-loop behavior the demo cares about and again prevented over-trusting validation loss.
- Continued scratch-CNN Lift for another 20 minutes from the previous final checkpoint: `runs/lift_vision_scratch_continue2_rollout_20260512`, 227 epochs, best val loss `0.4231` at epoch 151, final val `0.4454`.
- Second continuation rollout probes: 7/10 at resume, then 8/10 at epochs 50, 150, and 200; epoch 100 dipped to 7/10. The 10-start probes are useful trend checks but still noisy.
- Full 20-start eval after the second continuation: `last.pt` got 17/20 successes (`85%`), avg length `57.45`, failing `demo_1`, `demo_7`, and `demo_15`; `best.pt` got 13/20 (`65%`).
- Current high-level result: the scratch-CNN vision model is now close to the low-dim Lift baseline on the same 20 rendered starts, but still uses only 20 demos and a simple scratch visual encoder.
- README now surfaces the 17/20 vision result in the main results section, and `docs/vision-results.md` preserves the vision-run progression plus artifact paths.
