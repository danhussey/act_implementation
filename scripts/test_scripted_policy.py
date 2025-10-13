"""Test scripted policy execution."""

import sys
sys.path.insert(0, '.')

from act_implementation.envs.robosuite_wrapper import RoboSuiteWrapper
from act_implementation.envs.scripted_policy import ScriptedPickPlacePolicy
import numpy as np


def test_scripted_policy():
    """Test scripted policy for one episode."""
    print("Creating environment...")
    env = RoboSuiteWrapper(
        env_name="PickPlaceCan",
        robots="Panda",
        controller_name=None,
        camera_names=["agentview"],  # Use just one camera for speed
        camera_height=84,
        camera_width=84,
        horizon=500,
    )

    print("Creating scripted policy...")
    policy = ScriptedPickPlacePolicy(noise_scale=0.0)

    print("Resetting environment...")
    obs = env.reset()
    policy.reset()

    print(f"\nAction dimension: {env.action_dim}")
    print(f"Initial state: {policy.get_state_name()}")

    done = False
    step = 0
    max_steps = 300  # Limit steps for testing

    print("\nExecuting policy...")
    while not done and step < max_steps:
        # Get action from policy
        action, policy_done = policy.get_action(obs)

        if step % 10 == 0:
            print(f"Step {step}: state={policy.get_state_name()}, action_norm={np.linalg.norm(action):.3f}")

        # Take environment step
        obs, reward, env_done, info = env.step(action)

        done = policy_done or env_done
        step += 1

    print(f"\nFinished after {step} steps")
    print(f"Final state: {policy.get_state_name()}")
    print(f"Success: {info.get('success', False)}")

    env.close()


if __name__ == "__main__":
    test_scripted_policy()
