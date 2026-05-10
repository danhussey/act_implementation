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
- Latent sweep result on `demo_0`: 9 latent choices, 4 successes, success rate `44.4%`; artifacts in `runs/lift_mps_run_20260510/latent_sweep_demo0/`.
- Latent stress test: same sweep with `--scale 2.0` gave 3/9 successes, success rate `33.3%`; artifacts in `runs/lift_mps_run_20260510/latent_sweep_demo0_scale2/`.
- Latent sweep surprise: first predicted actions were very similar across latents, but rollout outcomes diverged. The latent effect appears to compound over the trajectory rather than showing as an obvious first-step mode switch.
