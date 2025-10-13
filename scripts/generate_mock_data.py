"""Generate mock demonstration data for testing the training pipeline."""

import h5py
import numpy as np
from pathlib import Path
import argparse


def generate_mock_episode(
    episode_length: int = 200,
    state_dim: int = 32,
    action_dim: int = 7,
    camera_height: int = 84,
    camera_width: int = 84,
    num_cameras: int = 2,
) -> dict:
    """
    Generate a single mock episode with realistic-looking data.

    The trajectories are synthetic but follow reasonable patterns:
    - States evolve smoothly with small changes
    - Actions are smooth and bounded
    - Images are random but consistent shape
    """
    # Generate smooth state trajectory
    states = np.zeros((episode_length, state_dim))
    states[0] = np.random.randn(state_dim) * 0.1
    for t in range(1, episode_length):
        # Random walk with small steps
        states[t] = states[t-1] + np.random.randn(state_dim) * 0.01

    # Generate smooth action trajectory
    actions = np.zeros((episode_length, action_dim))
    actions[0] = np.random.randn(action_dim) * 0.5
    for t in range(1, episode_length):
        # Smooth actions with momentum
        actions[t] = 0.9 * actions[t-1] + 0.1 * np.random.randn(action_dim) * 0.5
    # Clip to reasonable range
    actions = np.clip(actions, -1, 1)

    # Generate random images (in practice these would be rendered observations)
    images_agentview = np.random.randint(
        0, 256, size=(episode_length, 3, camera_height, camera_width), dtype=np.uint8
    )
    images_wrist = np.random.randint(
        0, 256, size=(episode_length, 3, camera_height, camera_width), dtype=np.uint8
    )

    # Generate rewards (gradually increasing to simulate task progress)
    rewards = np.linspace(0, 1, episode_length) + np.random.randn(episode_length) * 0.1

    return {
        "states": states.astype(np.float32),
        "actions": actions.astype(np.float32),
        "images_agentview": images_agentview,
        "images_robot0_eye_in_hand": images_wrist,
        "rewards": rewards.astype(np.float32),
        "success": np.random.rand() > 0.3,  # 70% success rate
    }


def generate_mock_dataset(
    output_path: Path,
    num_episodes: int = 50,
    episode_length: int = 200,
    state_dim: int = 32,
    action_dim: int = 7,
):
    """Generate a complete mock dataset."""
    print(f"Generating {num_episodes} mock episodes...")

    episodes = []
    for i in range(num_episodes):
        episode = generate_mock_episode(
            episode_length=episode_length,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        episodes.append(episode)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_episodes} episodes")

    # Save to HDF5
    print(f"Saving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        for i, episode in enumerate(episodes):
            group = f.create_group(f"episode_{i}")

            for key, value in episode.items():
                # Only compress arrays, not scalars
                if np.isscalar(value):
                    group.create_dataset(key, data=value)
                else:
                    group.create_dataset(key, data=value, compression="gzip")

        # Store metadata
        f.attrs["num_episodes"] = len(episodes)
        f.attrs["num_successful"] = sum(ep["success"] for ep in episodes)

    print(f"✓ Saved {len(episodes)} episodes to {output_path}")
    print(f"  Success rate: {sum(ep['success'] for ep in episodes)}/{len(episodes)}")
    print(f"  Total timesteps: {sum(len(ep['states']) for ep in episodes)}")


def main():
    parser = argparse.ArgumentParser(description="Generate mock demonstration data")
    parser.add_argument("--output", type=str, default="data/mock_demos.hdf5", help="Output file path")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--episode-length", type=int, default=200, help="Steps per episode")
    parser.add_argument("--state-dim", type=int, default=32, help="State dimension")
    parser.add_argument("--action-dim", type=int, default=7, help="Action dimension")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_mock_dataset(
        output_path=output_path,
        num_episodes=args.episodes,
        episode_length=args.episode_length,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )


if __name__ == "__main__":
    main()
