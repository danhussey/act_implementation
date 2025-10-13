"""Collect demonstration trajectories using scripted policy."""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

from act_implementation.envs.robosuite_wrapper import RoboSuiteWrapper
from act_implementation.envs.scripted_policy import ScriptedPickPlacePolicy


def collect_episode(env: RoboSuiteWrapper, policy: ScriptedPickPlacePolicy) -> dict:
    """
    Collect a single episode using the scripted policy.

    Returns:
        Episode data dict with observations, actions, rewards
    """
    obs = env.reset()
    policy.reset()

    states = []
    images = {cam_name: [] for cam_name in env.camera_names}
    actions = []
    rewards = []

    done = False
    step = 0
    max_steps = 500

    while not done and step < max_steps:
        # Get action from scripted policy
        action, policy_done = policy.get_action(obs)

        # Store current observation
        states.append(obs["state"])
        for cam_name in env.camera_names:
            images[cam_name].append(obs["images"][cam_name])
        actions.append(action)

        # Take environment step
        obs, reward, env_done, info = env.step(action)
        rewards.append(reward)

        done = policy_done or env_done
        step += 1

    # Convert to numpy arrays
    episode_data = {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
    }

    # Add images
    for cam_name in env.camera_names:
        episode_data[f"images_{cam_name}"] = np.array(images[cam_name])

    # Check if episode was successful
    episode_data["success"] = info.get("success", False)
    episode_data["length"] = step

    return episode_data


def save_episodes_hdf5(episodes: list, output_path: Path):
    """
    Save collected episodes to HDF5 file.

    Format:
        /episode_0/states: (T, state_dim)
        /episode_0/actions: (T, action_dim)
        /episode_0/images_agentview: (T, C, H, W)
        /episode_0/images_robot0_eye_in_hand: (T, C, H, W)
        /episode_0/rewards: (T,)
        /episode_0/success: scalar
    """
    with h5py.File(output_path, "w") as f:
        for i, episode in enumerate(episodes):
            group = f.create_group(f"episode_{i}")

            for key, value in episode.items():
                if key != "length":
                    group.create_dataset(key, data=value, compression="gzip")

        # Store metadata
        f.attrs["num_episodes"] = len(episodes)
        f.attrs["num_successful"] = sum(ep["success"] for ep in episodes)


def main():
    parser = argparse.ArgumentParser(description="Collect demonstration data")
    parser.add_argument("--env", type=str, default="PickPlaceCan", help="Environment name")
    parser.add_argument("--robot", type=str, default="Panda", help="Robot type")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--output", type=str, default="data/demos.hdf5", help="Output file path")
    parser.add_argument("--camera-height", type=int, default=84, help="Camera image height")
    parser.add_argument("--camera-width", type=int, default=84, help="Camera image width")
    parser.add_argument("--noise", type=float, default=0.005, help="Action noise scale for diversity")
    parser.add_argument("--keep-failed", action="store_true", help="Keep failed episodes")
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    print(f"Creating environment: {args.env}")
    env = RoboSuiteWrapper(
        env_name=args.env,
        robots=args.robot,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        horizon=500,
    )

    # Initialize scripted policy
    policy = ScriptedPickPlacePolicy(noise_scale=args.noise)

    # Collect episodes
    print(f"Collecting {args.episodes} episodes...")
    episodes = []
    num_attempts = 0
    pbar = tqdm(total=args.episodes)

    while len(episodes) < args.episodes:
        num_attempts += 1
        episode = collect_episode(env, policy)

        # Filter failed episodes unless --keep-failed
        if args.keep_failed or episode["success"]:
            episodes.append(episode)
            pbar.update(1)
            pbar.set_description(
                f"Success rate: {len(episodes)}/{num_attempts} "
                f"({100*len(episodes)/num_attempts:.1f}%)"
            )

    pbar.close()
    env.close()

    # Save to HDF5
    print(f"Saving {len(episodes)} episodes to {output_path}")
    save_episodes_hdf5(episodes, output_path)

    # Print statistics
    episode_lengths = [ep["length"] for ep in episodes]
    total_rewards = [ep["rewards"].sum() for ep in episodes]

    print(f"\nDataset Statistics:")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Successful episodes: {sum(ep['success'] for ep in episodes)}")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Total timesteps: {sum(episode_lengths)}")


if __name__ == "__main__":
    main()
