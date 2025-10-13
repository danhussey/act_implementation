"""Debug scripted policy to see why it's stuck."""

import sys
sys.path.insert(0, '.')

from act_implementation.envs.robosuite_wrapper import RoboSuiteWrapper
from act_implementation.envs.scripted_policy import ScriptedPickPlacePolicy
import numpy as np


def debug_policy():
    """Debug scripted policy."""
    env = RoboSuiteWrapper(
        env_name="PickPlaceCan",
        robots="Panda",
        camera_names=["agentview"],
        camera_height=84,
        camera_width=84,
    )

    policy = ScriptedPickPlacePolicy(noise_scale=0.0)

    obs = env.reset()
    policy.reset()

    raw_obs = obs["raw_obs"]

    print("Available keys in raw_obs:")
    for key in sorted(raw_obs.keys()):
        val = raw_obs[key]
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {type(val)}")

    print("\nEnd-effector position:", raw_obs["robot0_eef_pos"])

    if "Can_pos" in raw_obs:
        print("Can position:", raw_obs["Can_pos"])
        target = raw_obs["Can_pos"].copy()
        target[2] += policy.approach_height
        print("Target (approach) position:", target)
        print("Distance to target:", np.linalg.norm(raw_obs["robot0_eef_pos"] - target))

    # Execute a few steps
    print("\nExecuting 5 steps:")
    for i in range(5):
        action, done = policy.get_action(obs)
        print(f"\nStep {i}:")
        print(f"  Action: {action}")
        print(f"  EEF pos: {obs['raw_obs']['robot0_eef_pos']}")

        obs, reward, _, info = env.step(action)
        print(f"  New EEF pos: {obs['raw_obs']['robot0_eef_pos']}")
        print(f"  Movement: {np.linalg.norm(obs['raw_obs']['robot0_eef_pos'] - raw_obs['robot0_eef_pos'])}")
        raw_obs = obs["raw_obs"]

    env.close()


if __name__ == "__main__":
    debug_policy()
