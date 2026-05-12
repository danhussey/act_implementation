# Vision Results

Concise notes for the high-dimensional Lift path. These runs use
`agentview_image` plus robot proprioception and exclude the privileged
robomimic `object` state.

## Lift Scratch CNN

| Run | Checkpoint | Rollout eval | Notes |
| --- | --- | ---: | --- |
| `runs/lift_vision_scratch_20demos_20260512` | `best.pt` | 10/20 | First 20-epoch smoke on 20 rendered demos. |
| `runs/lift_vision_scratch_20min_20260512` | `best.pt` | 5/20 | Lowest validation loss was a poor actor. |
| `runs/lift_vision_scratch_20min_20260512` | `last.pt` | 9/20 | Final checkpoint beat validation-best. |
| `runs/lift_vision_scratch_continue_rollout_20260512` | `best.pt` | 12/20 | Added in-training rollout probes. |
| `runs/lift_vision_scratch_continue_rollout_20260512` | `last.pt` | 14/20 | First continuation improved closed-loop behavior. |
| `runs/lift_vision_scratch_continue2_rollout_20260512` | `best.pt` | 13/20 | Validation-best still lagged final behavior. |
| `runs/lift_vision_scratch_continue2_rollout_20260512` | `last.pt` | 17/20 | Best vision result so far; failures: `demo_1`, `demo_7`, `demo_15`. |

## Takeaways

- Rollout success improved from 10/20 to 17/20 with longer scratch-CNN training.
- Validation loss remained a weak selector: `last.pt` repeatedly beat `best.pt`
  in closed-loop evaluation.
- The current vision policy is close to the low-dim Lift baseline, but the
  baseline still has a small edge at 18/20 and uses privileged object state.
- The next clean comparison is frozen pretrained ResNet-18 versus scratch CNN on
  the same rendered 20-demo file.

## Useful Artifacts

- Latest rollout metrics:
  `runs/lift_vision_scratch_continue2_rollout_20260512/eval_20_z0_last/eval_metrics.json`
- Latest rollout MP4s:
  `runs/lift_vision_scratch_continue2_rollout_20260512/eval_20_z0_last/videos/`
- Latest rollout curve:
  `runs/lift_vision_scratch_continue2_rollout_20260512/rollout_curve.svg`
- Rendered image dataset:
  `data/lift_ph_agentview_20demos.hdf5`
