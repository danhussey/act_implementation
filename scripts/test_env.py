"""Test script to verify RoboSuite environment wrapper."""

import sys
sys.path.insert(0, '.')

from act_implementation.envs.robosuite_wrapper import RoboSuiteWrapper
import numpy as np


def test_environment():
    """Test basic environment functionality."""
    print("Creating environment...")
    env = RoboSuiteWrapper(
        env_name="PickPlaceCan",
        robots="Panda",
        controller_name=None,  # Use robot's default controller
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_height=84,
        camera_width=84,
        horizon=500,
    )

    print(f"Action dimension: {env.action_dim}")
    print(f"State dimension: {env.state_dim}")
    print(f"Camera names: {env.camera_names}")

    print("\nResetting environment...")
    obs = env.reset()
    print(f"State shape: {obs['state'].shape}")
    print(f"Number of cameras: {len(obs['images'])}")
    for cam_name, img in obs['images'].items():
        print(f"  {cam_name} shape: {img.shape}")

    print("\nTaking random step...")
    action = np.random.randn(env.action_dim) * 0.1
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}")
    print(f"Done: {done}")

    env.close()
    print("\nEnvironment test passed!")


if __name__ == "__main__":
    test_environment()
