"""Parallel demonstration collection using multiprocessing for speedup."""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Dict
import tempfile
import shutil

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


def worker_collect_episodes(args: tuple) -> str:
    """
    Worker function to collect episodes in a separate process.

    Args:
        args: Tuple of (worker_id, num_episodes, config_dict, temp_dir)

    Returns:
        Path to temporary HDF5 file with collected episodes
    """
    worker_id, num_episodes, config, temp_dir = args

    # Create environment for this worker
    env = RoboSuiteWrapper(
        env_name=config["env"],
        robots=config["robot"],
        camera_names=config["camera_names"],
        camera_height=config["camera_height"],
        camera_width=config["camera_width"],
        horizon=config["horizon"],
    )

    # Create policy
    policy = ScriptedPickPlacePolicy(noise_scale=config["noise"])

    # Collect episodes
    episodes = []
    num_attempts = 0

    pbar = tqdm(
        total=num_episodes,
        position=worker_id,
        desc=f"Worker {worker_id}",
        leave=True
    )

    while len(episodes) < num_episodes:
        num_attempts += 1
        episode = collect_episode(env, policy)

        # Filter failed episodes unless keep_failed
        if config["keep_failed"] or episode["success"]:
            episodes.append(episode)
            pbar.update(1)
            success_rate = len(episodes) / num_attempts
            pbar.set_description(
                f"Worker {worker_id} ({success_rate:.1%} success)"
            )

    pbar.close()
    env.close()

    # Save episodes to temporary file
    temp_file = Path(temp_dir) / f"worker_{worker_id}.hdf5"
    with h5py.File(temp_file, "w") as f:
        for i, episode in enumerate(episodes):
            group = f.create_group(f"episode_{i}")
            for key, value in episode.items():
                if key != "length":
                    if np.isscalar(value):
                        group.create_dataset(key, data=value)
                    else:
                        group.create_dataset(key, data=value, compression="gzip")

        # Store metadata
        f.attrs["num_episodes"] = len(episodes)
        f.attrs["num_successful"] = sum(ep["success"] for ep in episodes)

    return str(temp_file)


def merge_hdf5_files(temp_files: List[str], output_path: Path):
    """
    Merge multiple temporary HDF5 files into final output file.

    Args:
        temp_files: List of paths to temporary HDF5 files
        output_path: Final output path
    """
    print(f"\nMerging {len(temp_files)} temporary files into {output_path}...")

    all_episodes = []
    total_successful = 0

    # Load all episodes from temp files
    for temp_file in temp_files:
        with h5py.File(temp_file, "r") as f:
            num_episodes = f.attrs["num_episodes"]
            total_successful += f.attrs["num_successful"]

            for i in range(num_episodes):
                episode_data = {}
                group = f[f"episode_{i}"]
                for key in group.keys():
                    episode_data[key] = group[key][()]
                all_episodes.append(episode_data)

    # Write merged file
    with h5py.File(output_path, "w") as f:
        for i, episode in enumerate(all_episodes):
            group = f.create_group(f"episode_{i}")
            for key, value in episode.items():
                if np.isscalar(value):
                    group.create_dataset(key, data=value)
                else:
                    group.create_dataset(key, data=value, compression="gzip")

        # Store metadata
        f.attrs["num_episodes"] = len(all_episodes)
        f.attrs["num_successful"] = total_successful

    print(f"✓ Merged {len(all_episodes)} episodes")
    print(f"  Success rate: {total_successful}/{len(all_episodes)} "
          f"({100*total_successful/len(all_episodes):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Collect demonstration data in parallel using multiprocessing"
    )
    parser.add_argument("--env", type=str, default="PickPlaceCan", help="Environment name")
    parser.add_argument("--robot", type=str, default="Panda", help="Robot type")
    parser.add_argument("--episodes", type=int, default=100, help="Total episodes to collect")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="data/demos.hdf5", help="Output file path")
    parser.add_argument("--camera-height", type=int, default=84, help="Camera image height")
    parser.add_argument("--camera-width", type=int, default=84, help="Camera image width")
    parser.add_argument("--horizon", type=int, default=500, help="Max episode length")
    parser.add_argument("--noise", type=float, default=0.005, help="Action noise scale")
    parser.add_argument("--keep-failed", action="store_true", help="Keep failed episodes")
    args = parser.parse_args()

    # Validate arguments
    if args.episodes < args.workers:
        print(f"Warning: episodes ({args.episodes}) < workers ({args.workers})")
        print(f"Reducing workers to {args.episodes}")
        args.workers = args.episodes

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Distribute episodes across workers
    episodes_per_worker = args.episodes // args.workers
    extra_episodes = args.episodes % args.workers

    worker_episodes = [episodes_per_worker] * args.workers
    for i in range(extra_episodes):
        worker_episodes[i] += 1

    print(f"Collecting {args.episodes} episodes using {args.workers} parallel workers")
    print(f"Episodes per worker: {worker_episodes}")

    # Create config dict for workers
    config = {
        "env": args.env,
        "robot": args.robot,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_height": args.camera_height,
        "camera_width": args.camera_width,
        "horizon": args.horizon,
        "noise": args.noise,
        "keep_failed": args.keep_failed,
    }

    # Create temporary directory for worker outputs
    temp_dir = tempfile.mkdtemp(prefix="robosuite_parallel_")

    try:
        # Prepare worker arguments
        worker_args = [
            (i, worker_episodes[i], config, temp_dir)
            for i in range(args.workers)
        ]

        # Use spawn method to avoid memory leaks (important for RoboSuite)
        mp.set_start_method('spawn', force=True)

        # Run parallel collection
        with mp.Pool(processes=args.workers) as pool:
            temp_files = pool.map(worker_collect_episodes, worker_args)

        # Merge results
        merge_hdf5_files(temp_files, output_path)

        # Print final statistics
        with h5py.File(output_path, "r") as f:
            num_episodes = f.attrs["num_episodes"]
            num_successful = f.attrs["num_successful"]

            episode_lengths = []
            total_rewards = []
            for i in range(num_episodes):
                group = f[f"episode_{i}"]
                episode_lengths.append(len(group["states"]))
                total_rewards.append(group["rewards"][()].sum())

            print(f"\nFinal Dataset Statistics:")
            print(f"  Total episodes: {num_episodes}")
            print(f"  Successful episodes: {num_successful}")
            print(f"  Success rate: {100*num_successful/num_episodes:.1f}%")
            print(f"  Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
            print(f"  Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
            print(f"  Total timesteps: {sum(episode_lengths)}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n✓ Cleaned up temporary files")


if __name__ == "__main__":
    main()
