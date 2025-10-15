"""Test manual grasping with different gripper strategies."""

import sys
sys.path.insert(0, '.')

import numpy as np
import robosuite as suite


def test_grasp_strategies():
    """Try different grasping strategies to find what works."""
    env = suite.make(
        env_name="PickPlaceCan",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,
        control_freq=50,
    )

    print("="*60)
    print("STRATEGY 1: Grasp from side with gripper horizontal")
    print("="*60)

    obs = env.reset()
    can_pos = obs['Can_pos']
    print(f"Can position: {can_pos}")

    # Position gripper to side of can, fingers vertical to grasp
    target = can_pos.copy()
    target[2] = can_pos[2] + 0.06  # At can height
    target[0] -= 0.08  # Approach from side

    print(f"\nMoving to side of can at {target}...")

    # Open gripper wide first
    for i in range(20):
        action = np.zeros(7)
        action[6] = -1.0  # Try "closing" command (which seems to open based on previous test)
        obs, _, _, _ = env.step(action)

    print(f"Gripper opened: {obs['robot0_gripper_qpos']}")

    # Move to position
    for i in range(100):
        eef_pos = obs['robot0_eef_pos']
        delta = target - eef_pos
        distance = np.linalg.norm(delta)

        if distance < 0.015:
            break

        max_delta = 0.03
        if distance > max_delta:
            delta = delta / distance * max_delta
        normalized_delta = delta / 0.05

        action = np.zeros(7)
        action[:3] = normalized_delta
        action[6] = -1.0  # Keep open

        obs, _, _, _ = env.step(action)

    print(f"Reached side position. EEF: {obs['robot0_eef_pos']}")

    # Move toward can (into grasping position)
    for i in range(50):
        action = np.zeros(7)
        action[0] = 0.5  # Move in +X toward can
        action[6] = -1.0  # Keep open
        obs, _, _, _ = env.step(action)

        eef_pos = obs['robot0_eef_pos']
        can_pos = obs['Can_pos']
        dist = np.linalg.norm(eef_pos[:2] - can_pos[:2])

        if i % 10 == 0:
            print(f"Step {i}: dist to can (XY) = {dist:.3f}m")

        if dist < 0.06:  # Fingers around can
            print(f"Can between fingers at step {i}")
            break

    # Close gripper
    print("\nClosing gripper...")
    for i in range(40):
        action = np.zeros(7)
        action[6] = 1.0  # "Open" command (which seems to close)
        obs, _, _, _ = env.step(action)
        if i % 10 == 0:
            print(f"Step {i}: gripper={obs['robot0_gripper_qpos']}")

    # Lift
    print("\nLifting...")
    initial_can_z = obs['Can_pos'][2]
    for i in range(50):
        action = np.zeros(7)
        action[2] = 1.0
        action[6] = 1.0  # Keep closed
        obs, _, _, _ = env.step(action)

        if i % 10 == 0:
            can_z = obs['Can_pos'][2]
            print(f"Step {i}: Can Z = {can_z:.3f} (delta: {can_z - initial_can_z:.3f})")

    z_lift = obs['Can_pos'][2] - initial_can_z
    print(f"\n✓ Can lifted: {z_lift:.3f}m" if z_lift > 0.05 else f"\n✗ Failed: {z_lift:.3f}m")

    env.close()

    # STRATEGY 2: Try using actual gripper command directly
    print("\n" + "="*60)
    print("STRATEGY 2: Grasp from above with HOLD mode")
    print("="*60)

    env = suite.make(
        env_name="PickPlaceCan",
        robots="Panda",
        has_renderer=False,
        use_camera_obs=False,
        horizon=500,
        control_freq=50,
    )

    obs = env.reset()
    can_pos = obs['Can_pos']

    # Move directly above can with gripper wide open
    target = can_pos.copy()
    target[2] += 0.15

    # Open maximally
    for i in range(30):
        action = np.zeros(7)
        action[6] = -1.0
        obs, _, _, _ = env.step(action)

    print(f"Max open gripper: {obs['robot0_gripper_qpos']}")

    # Move above
    for i in range(80):
        eef_pos = obs['robot0_eef_pos']
        delta = target - eef_pos
        distance = np.linalg.norm(delta)
        if distance < 0.02:
            break

        normalized_delta = np.clip(delta / 0.05, -1, 1)
        action = np.zeros(7)
        action[:3] = normalized_delta
        action[6] = -1.0
        obs, _, _, _ = env.step(action)

    # Descend slowly onto can with gripper open
    target[2] = can_pos[2] + 0.03
    for i in range(60):
        eef_pos = obs['robot0_eef_pos']
        delta = target - eef_pos

        action = np.zeros(7)
        action[:3] = np.clip(delta / 0.05, -1, 1) * 0.3  # Slow descent
        action[6] = -1.0  # Open
        obs, _, _, _ = env.step(action)

        if i % 15 == 0:
            print(f"Descending: EEF_z={eef_pos[2]:.3f}, Can_z={obs['Can_pos'][2]:.3f}")

    print(f"\nAt grasp height. Gripper: {obs['robot0_gripper_qpos']}")

    # Close gripper firmly
    print("Closing gripper...")
    for i in range(50):
        action = np.zeros(7)
        action[6] = 1.0
        obs, _, _, _ = env.step(action)
        if i % 10 == 0:
            print(f"  Step {i}: {obs['robot0_gripper_qpos']}")

    # Lift with closed gripper
    initial_can_z = obs['Can_pos'][2]
    print(f"\nLifting from z={initial_can_z:.3f}...")
    for i in range(60):
        action = np.zeros(7)
        action[2] = 0.8
        action[6] = 1.0
        obs, _, _, _ = env.step(action)

        if i % 15 == 0:
            print(f"  Step {i}: Can_z={obs['Can_pos'][2]:.3f}")

    z_lift = obs['Can_pos'][2] - initial_can_z
    print(f"\n{'✓ SUCCESS' if z_lift > 0.05 else '✗ FAILED'}: Can lifted {z_lift:.3f}m")

    env.close()


if __name__ == "__main__":
    test_grasp_strategies()
