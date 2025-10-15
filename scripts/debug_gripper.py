"""Debug gripper control and can grasping."""

import sys
sys.path.insert(0, '.')

import numpy as np
import robosuite as suite


def test_gripper_control():
    """Test gripper opening/closing and can interaction."""
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
    print("TESTING GRIPPER CONTROL")
    print("="*60)

    obs = env.reset()

    print(f"Initial gripper state: {obs['robot0_gripper_qpos']}")
    print(f"Initial EEF pos: {obs['robot0_eef_pos']}")
    print(f"Initial Can pos: {obs['Can_pos']}")

    # Test 1: Open gripper
    print("\n--- Test 1: Opening gripper ---")
    for i in range(20):
        action = np.zeros(7)
        action[6] = 1.0  # Open gripper
        obs, reward, done, info = env.step(action)
        if i % 5 == 0:
            print(f"Step {i}: gripper_qpos={obs['robot0_gripper_qpos']}")

    # Test 2: Close gripper
    print("\n--- Test 2: Closing gripper ---")
    for i in range(20):
        action = np.zeros(7)
        action[6] = -1.0  # Close gripper
        obs, reward, done, info = env.step(action)
        if i % 5 == 0:
            print(f"Step {i}: gripper_qpos={obs['robot0_gripper_qpos']}")

    # Test 3: Move to can and try to grasp
    print("\n--- Test 3: Moving to can and grasping ---")
    obs = env.reset()
    can_pos = obs['Can_pos']
    eef_pos = obs['robot0_eef_pos']

    print(f"Can position: {can_pos}")
    print(f"EEF position: {eef_pos}")
    print(f"Distance to can: {np.linalg.norm(can_pos - eef_pos):.3f}m")

    # Move above can
    print("\nPhase 1: Moving above can...")
    target = can_pos.copy()
    target[2] += 0.15  # 15cm above

    for i in range(100):
        delta = target - eef_pos
        distance = np.linalg.norm(delta)

        if distance < 0.02:
            print(f"Reached above can at step {i}")
            break

        # Normalize to [-1, 1] range for controller
        max_delta = 0.05
        if distance > max_delta:
            delta = delta / distance * max_delta
        normalized_delta = delta / 0.05

        action = np.zeros(7)
        action[:3] = normalized_delta
        action[6] = 1.0  # Keep gripper open

        obs, reward, done, info = env.step(action)
        eef_pos = obs['robot0_eef_pos']

        if i % 20 == 0:
            print(f"Step {i}: dist={distance:.3f}, EEF={eef_pos}")

    # Move down to can
    print("\nPhase 2: Moving down to can...")
    target = can_pos.copy()
    target[2] += 0.02  # Just above can surface

    for i in range(100):
        delta = target - eef_pos
        distance = np.linalg.norm(delta)

        if distance < 0.02:
            print(f"Reached can at step {i}")
            break

        max_delta = 0.05
        if distance > max_delta:
            delta = delta / distance * max_delta
        normalized_delta = delta / 0.05

        action = np.zeros(7)
        action[:3] = normalized_delta
        action[6] = 1.0  # Keep gripper open initially

        # Start closing when very close
        if distance < 0.05:
            action[6] = -1.0

        obs, reward, done, info = env.step(action)
        eef_pos = obs['robot0_eef_pos']
        can_pos = obs['Can_pos']

        if i % 10 == 0:
            print(f"Step {i}: dist={distance:.3f}, EEF={eef_pos}, Can={can_pos}, gripper={obs['robot0_gripper_qpos']}")

    # Close gripper fully
    print("\nPhase 3: Closing gripper...")
    for i in range(30):
        action = np.zeros(7)
        action[6] = -1.0
        obs, reward, done, info = env.step(action)
        if i % 5 == 0:
            print(f"Step {i}: gripper={obs['robot0_gripper_qpos']}, Can={obs['Can_pos']}")

    # Lift up
    print("\nPhase 4: Lifting up...")
    initial_can_z = obs['Can_pos'][2]
    for i in range(50):
        action = np.zeros(7)
        action[2] = 1.0  # Move up
        action[6] = -1.0  # Keep gripper closed
        obs, reward, done, info = env.step(action)
        can_z = obs['Can_pos'][2]
        eef_z = obs['robot0_eef_pos'][2]

        if i % 10 == 0:
            print(f"Step {i}: EEF_z={eef_z:.3f}, Can_z={can_z:.3f}, delta_z={can_z - initial_can_z:.3f}")

    final_can_z = obs['Can_pos'][2]
    z_lift = final_can_z - initial_can_z

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Can lifted by: {z_lift:.3f}m")
    if z_lift > 0.05:
        print("✓ SUCCESS: Can was grasped and lifted!")
    else:
        print("✗ FAILURE: Can did not lift (gripper not grasping)")

    env.close()


if __name__ == "__main__":
    test_gripper_control()
