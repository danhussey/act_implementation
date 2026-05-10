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
