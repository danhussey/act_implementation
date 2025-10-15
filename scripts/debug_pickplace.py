"""Debug script to understand PickPlaceCan environment and find target positions."""

import sys
sys.path.insert(0, '.')

import numpy as np
import robosuite as suite


def debug_pickplace_env():
    """Explore the PickPlaceCan environment to understand observations and targets."""
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
    print("ENVIRONMENT EXPLORATION")
    print("="*60)

    # Reset and get initial observation
    obs = env.reset()

    print("\nAvailable observation keys:")
    for key in sorted(obs.keys()):
        value = obs[key]
        if isinstance(value, np.ndarray):
            print(f"  {key:30s} shape: {value.shape} dtype: {value.dtype}")
            if value.size <= 10:
                print(f"    -> {value}")

    # Print key positions
    print("\n" + "="*60)
    print("KEY POSITIONS")
    print("="*60)
    print(f"Robot EEF position: {obs['robot0_eef_pos']}")
    print(f"Can position:       {obs['Can_pos']}")

    # Check for target/bin position
    target_keys = [k for k in obs.keys() if 'bin' in k.lower() or 'target' in k.lower() or 'goal' in k.lower()]
    print(f"\nTarget-related keys: {target_keys}")
    for key in target_keys:
        print(f"  {key}: {obs[key]}")

    # Get environment info
    print("\n" + "="*60)
    print("ENVIRONMENT INFO")
    print("="*60)
    print(f"Action dimension: {env.action_dim}")
    print(f"Action spec: {env.action_spec}")

    # Check the actual task placement target from the environment
    # In RoboSuite, PickPlaceCan has a bin/placement location
    if hasattr(env, 'table_offset'):
        print(f"Table offset: {env.table_offset}")
    if hasattr(env, 'bin_pos'):
        print(f"Bin position: {env.bin_pos}")
    if hasattr(env, 'placement_initializer'):
        print(f"Placement initializer: {env.placement_initializer}")

    # Try to get the bin location directly
    print("\n" + "="*60)
    print("SEARCHING FOR BIN/TARGET LOCATION")
    print("="*60)

    # Look for bin-related bodies in MuJoCo model
    print("Bodies in MuJoCo model containing 'bin':")
    bin_pos = None
    for i in range(env.sim.model.nbody):
        body_name = env.sim.model.body_id2name(i)
        if body_name and 'bin' in body_name.lower():
            pos = env.sim.data.body_xpos[i]
            print(f"  {body_name}: {pos}")
            if bin_pos is None:
                bin_pos = pos

    # If no bin found, check for any placement-related bodies
    if bin_pos is None:
        print("\nLooking for placement-related bodies:")
        for i in range(env.sim.model.nbody):
            body_name = env.sim.model.body_id2name(i)
            if body_name and any(keyword in body_name.lower() for keyword in ['placement', 'target', 'goal']):
                pos = env.sim.data.body_xpos[i]
                print(f"  {body_name}: {pos}")
                if bin_pos is None:
                    bin_pos = pos

    if bin_pos is None:
        print("\nNo explicit bin/target body found. Using default target location.")
        # Default for PickPlaceCan based on RoboSuite conventions
        bin_pos = np.array([0.0, 0.3, 0.8])

    # Check which bin is the target by looking at task configuration
    print("\n" + "="*60)
    print("CHECKING TASK TARGET")
    print("="*60)

    # The target bin is determined by the task - let's check the placement sampler
    if hasattr(env, 'placement_initializer'):
        print(f"Placement initializer type: {type(env.placement_initializer)}")
        if hasattr(env.placement_initializer, 'samplers'):
            print(f"Number of samplers: {len(env.placement_initializer.samplers)}")
            for idx, sampler in enumerate(env.placement_initializer.samplers):
                print(f"  Sampler {idx}: {type(sampler).__name__}")
                if hasattr(sampler, 'x_range'):
                    print(f"    x_range: {sampler.x_range}")
                if hasattr(sampler, 'y_range'):
                    print(f"    y_range: {sampler.y_range}")

    # Get all bin positions
    all_bins = []
    for i in range(env.sim.model.nbody):
        body_name = env.sim.model.body_id2name(i)
        if body_name and 'bin' in body_name.lower():
            pos = env.sim.data.body_xpos[i]
            all_bins.append((body_name, pos.copy()))

    print(f"\nAll bins: {all_bins}")

    # Test which bin gives success by manually placing can there
    print("\n" + "="*60)
    print("TESTING PLACEMENT IN EACH BIN")
    print("="*60)

    for bin_name, bin_pos in all_bins:
        # Reset environment
        obs = env.reset()
        can_id = env.sim.model.body_name2id("Can")

        # Teleport can to bin (for testing)
        test_pos = bin_pos.copy()
        test_pos[2] += 0.05  # Slightly above bin
        env.sim.data.set_joint_qpos("Can_joint0", np.concatenate([test_pos, obs['Can_quat']]))
        env.sim.forward()

        # Take a step and check reward
        obs, reward, done, info = env.step(np.zeros(env.action_dim))
        print(f"\n{bin_name} at {bin_pos}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  Success: {info.get('success', False)}")

    env.close()
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR POLICY")
    print("="*60)
    print(f"✓ Target position should be set to: {bin_pos}")
    print(f"✓ This is where the can needs to be placed")


if __name__ == "__main__":
    debug_pickplace_env()
