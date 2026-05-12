# Vision Results

Concise notes for the high-dimensional Lift path. These runs use
`agentview_image` plus robot proprioception and exclude the privileged
robomimic `object` state. `agentview` is a third-person camera, not a wrist/POV
camera, so it can still suffer from hand/object occlusion.

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

## Lift Frozen Pretrained ResNet-18

| Run | Checkpoint | Rollout eval | Notes |
| --- | --- | ---: | --- |
| `runs/lift_vision_resnet18_frozen_20min_20260512` | `best.pt` | 16/20 | Best validation checkpoint at epoch 90; failures: `demo_0`, `demo_5`, `demo_9`, `demo_18`. |
| `runs/lift_vision_resnet18_frozen_20min_20260512` | `last.pt` | 16/20 | Final checkpoint matched success count but failed different starts: `demo_2`, `demo_3`, `demo_7`, `demo_13`. |

In-loop rollout probes were noisy but informative: 0/10 before training, 9/10
at epoch 50, 8/10 at epochs 100, 150, 200, and 250, then 5/10 at epoch 300.
The full eval did not collapse as far as the epoch-300 probe suggested, but the
late probe still caught a real degradation risk.

## Takeaways

- Rollout success improved from 10/20 to 17/20 with longer scratch-CNN training.
- Validation loss remained a weak selector: `last.pt` repeatedly beat `best.pt`
  in closed-loop evaluation.
- The current vision policy is close to the low-dim Lift baseline, but the
  baseline still has a small edge at 18/20 and uses privileged object state.
- Frozen pretrained ResNet-18 reached strong behavior quickly but did not beat
  the longer scratch-CNN run on the full 20-start eval.
- Observation media now avoids duplicating the vision panel when the rollout
  view and policy camera are both `agentview`. The low-dim media remains
  side-by-side because the policy input is spatial simulator state, not pixels.

## Useful Artifacts

- Latest rollout metrics:
  `runs/lift_vision_scratch_continue2_rollout_20260512/eval_20_z0_last/eval_metrics.json`
- Pretrained ResNet rollout metrics:
  `runs/lift_vision_resnet18_frozen_20min_20260512/eval_20_z0_best/eval_metrics.json`
- Latest rollout MP4s:
  `runs/lift_vision_scratch_continue2_rollout_20260512/eval_20_z0_last/videos/`
- Observation GIFs:
  `docs/assets/lift_lowdim_policy_view_demo10.gif`,
  `docs/assets/lift_scratchcnn_policy_view_demo10.gif`,
  `docs/assets/lift_resnet18_policy_view_demo10.gif`
- Latest rollout curve:
  `runs/lift_vision_scratch_continue2_rollout_20260512/rollout_curve.svg`
- Pretrained rollout curve:
  `runs/lift_vision_resnet18_frozen_20min_20260512/rollout_curve.svg`
- Rendered image dataset:
  `data/lift_ph_agentview_20demos.hdf5`
