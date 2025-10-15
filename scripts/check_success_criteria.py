"""Check RoboSuite PickPlaceCan success criteria."""

import sys
sys.path.insert(0, '.')

import numpy as np
import robosuite as suite


def check_success_criteria():
    """Understand what makes the PickPlaceCan task successful."""
    print("Creating PickPlaceCan environment...")
    env = suite.make(
        env_name="PickPlaceCan",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,
        control_freq=50,
    )

    print("\n" + "="*60)
    print("CHECKING SUCCESS/REWARD FUNCTION")
    print("="*60)

    # Check the reward function
    if hasattr(env, 'reward'):
        print(f"Reward function: {env.reward}")

    # Reset and check initial reward
    obs = env.reset()
    obs, reward, done, info = env.step(np.zeros(env.action_dim))

    print(f"\nInitial reward: {reward}")
    print(f"Initial success: {info.get('success', 'N/A')}")
    print(f"Info keys: {list(info.keys())}")

    # Get positions
    can_pos = obs["Can_pos"]
    print(f"\nCan position: {can_pos}")

    # Get bin positions
    bin1_id = env.sim.model.body_name2id("bin1")
    bin2_id = env.sim.model.body_name2id("bin2")
    bin1_pos = env.sim.data.body_xpos[bin1_id]
    bin2_pos = env.sim.data.body_xpos[bin2_id]

    print(f"Bin1 position: {bin1_pos}")
    print(f"Bin2 position: {bin2_pos}")

    # Try manually placing can in bin and check success
    print("\n" + "="*60)
    print("TESTING MANUAL CAN PLACEMENT")
    print("="*60)

    # Test placing can in bin1
    obs = env.reset()
    can_quat = obs["Can_quat"]

    # Place can in center of bin1, slightly above
    test_pos = bin1_pos.copy()
    test_pos[2] += 0.05

    # Set can position
    can_joint_name = "Can_joint0"
    env.sim.data.set_joint_qpos(can_joint_name, np.concatenate([test_pos, can_quat]))
    env.sim.forward()

    # Take a step
    for i in range(20):
        obs, reward, done, info = env.step(np.zeros(env.action_dim))
        can_pos = obs["Can_pos"]
        print(f"Step {i}: Can={can_pos}, Reward={reward:.3f}, Success={info.get('success', False)}")
        if info.get('success', False):
            break

    env.close()


if __name__ == "__main__":
    check_success_criteria()
