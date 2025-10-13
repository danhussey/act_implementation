"""Evaluation script for ACT policy with temporal ensembling."""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

from act_implementation.models.vision_encoder import VisionEncoder
from act_implementation.models.act_model import ACTModel
from act_implementation.envs.robosuite_wrapper import RoboSuiteWrapper
from act_implementation.data.dataset import DemonstrationDataset


class TemporalEnsemblePolicy:
    """
    Policy wrapper that performs temporal ensembling of action chunks.

    Maintains a buffer of overlapping action predictions and ensembles them
    for smoother and more consistent actions.
    """

    def __init__(
        self,
        vision_encoder: torch.nn.Module,
        act_model: torch.nn.Module,
        dataset: DemonstrationDataset,
        device: torch.device,
        chunk_size: int = 10,
        num_samples: int = 1,
        temporal_ensemble: bool = True,
    ):
        """
        Initialize temporal ensemble policy.

        Args:
            vision_encoder: Vision encoder model
            act_model: ACT model
            dataset: Dataset (for normalization stats)
            device: Device to run on
            chunk_size: Action chunk size
            num_samples: Number of CVAE samples to average
            temporal_ensemble: Whether to use temporal ensembling
        """
        self.vision_encoder = vision_encoder
        self.act_model = act_model
        self.dataset = dataset
        self.device = device
        self.chunk_size = chunk_size
        self.num_samples = num_samples
        self.temporal_ensemble = temporal_ensemble

        # Temporal ensemble buffer
        self.action_buffer = []
        self.buffer_size = chunk_size

        self.vision_encoder.eval()
        self.act_model.eval()

    def reset(self):
        """Reset the policy (clear buffer)."""
        self.action_buffer = []

    def get_action(self, obs: dict) -> np.ndarray:
        """
        Get action from policy with temporal ensembling.

        Args:
            obs: Observation dict from environment

        Returns:
            action: Single action to execute
        """
        with torch.no_grad():
            # Prepare observation
            state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(self.device)
            images = {
                k: torch.from_numpy(v).float().unsqueeze(0).to(self.device)
                for k, v in obs["images"].items()
            }

            # Normalize
            if self.dataset.normalize_states:
                state = (state - self.dataset.state_mean.to(self.device)) / self.dataset.state_std.to(self.device)

            # Get visual features
            visual_features = self.vision_encoder(images)

            # Get action chunk
            action_chunk = self.act_model.get_action(
                visual_features, state, num_samples=self.num_samples
            )  # (1, chunk_size, action_dim)

            # Unnormalize actions
            action_chunk = self.dataset.unnormalize_actions(action_chunk.cpu())
            action_chunk = action_chunk.squeeze(0).numpy()  # (chunk_size, action_dim)

            if self.temporal_ensemble:
                # Add to buffer
                self.action_buffer.append(action_chunk)

                # Keep buffer size limited
                if len(self.action_buffer) > self.buffer_size:
                    self.action_buffer.pop(0)

                # Ensemble: average the first action from all chunks in buffer
                # Weight more recent predictions higher
                weights = np.exp(np.linspace(0, 1, len(self.action_buffer)))
                weights = weights / weights.sum()

                ensembled_action = np.zeros(action_chunk.shape[1])
                for i, (chunk, weight) in enumerate(zip(self.action_buffer, weights)):
                    # Use the i-th action from this chunk (accounting for time offset)
                    action_idx = len(self.action_buffer) - 1 - i
                    if action_idx < len(chunk):
                        ensembled_action += weight * chunk[action_idx]

                return ensembled_action
            else:
                # No ensembling: just use first action from chunk
                return action_chunk[0]


def evaluate_policy(
    policy: TemporalEnsemblePolicy,
    env: RoboSuiteWrapper,
    num_episodes: int = 50,
    max_steps: int = 500,
) -> dict:
    """
    Evaluate policy in environment.

    Args:
        policy: Policy to evaluate
        env: Environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode

    Returns:
        Dictionary with evaluation metrics
    """
    success_count = 0
    episode_lengths = []
    episode_rewards = []

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        policy.reset()

        done = False
        step = 0
        total_reward = 0.0

        while not done and step < max_steps:
            # Get action from policy
            action = policy.get_action(obs)

            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

        # Check success
        if info.get("success", False):
            success_count += 1

        episode_lengths.append(step)
        episode_rewards.append(total_reward)

    return {
        "success_rate": success_count / num_episodes,
        "avg_episode_length": np.mean(episode_lengths),
        "avg_episode_reward": np.mean(episode_rewards),
        "std_episode_reward": np.std(episode_rewards),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset (for normalization)")
    parser.add_argument("--env", type=str, default="PickPlaceCan", help="Environment name")
    parser.add_argument("--robot", type=str, default="Panda", help="Robot type")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="mps", help="Device (cuda/mps/cpu)")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of CVAE samples")
    parser.add_argument("--no-temporal-ensemble", action="store_true", help="Disable temporal ensembling")
    parser.add_argument("--camera-height", type=int, default=84, help="Camera height")
    parser.add_argument("--camera-width", type=int, default=84, help="Camera width")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    # Load dataset for normalization stats
    print("Loading dataset...")
    dataset = DemonstrationDataset(
        data_path=args.data,
        chunk_size=config["chunk_size"],
        normalize_actions=True,
        normalize_states=True,
    )
    stats = dataset.get_stats()

    # Create models
    print("Creating models...")
    vision_encoder = VisionEncoder(
        backbone="resnet18",
        pretrained=True,
        num_cameras=len(dataset.camera_names),
        feature_dim=512,
    ).to(device)

    act_model = ACTModel(
        state_dim=stats["state_dim"],
        action_dim=stats["action_dim"],
        visual_feature_dim=512 * len(dataset.camera_names),
        chunk_size=config["chunk_size"],
        hidden_dim=config.get("hidden_dim", 512),
        num_encoder_layers=config.get("num_encoder_layers", 4),
        num_decoder_layers=config.get("num_decoder_layers", 4),
        latent_dim=config.get("latent_dim", 32),
    ).to(device)

    # Load weights
    vision_encoder.load_state_dict(checkpoint["vision_encoder"])
    act_model.load_state_dict(checkpoint["act_model"])

    # Create policy
    policy = TemporalEnsemblePolicy(
        vision_encoder=vision_encoder,
        act_model=act_model,
        dataset=dataset,
        device=device,
        chunk_size=config["chunk_size"],
        num_samples=args.num_samples,
        temporal_ensemble=not args.no_temporal_ensemble,
    )

    # Create environment
    print(f"Creating environment: {args.env}")
    env = RoboSuiteWrapper(
        env_name=args.env,
        robots=args.robot,
        camera_names=dataset.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        horizon=500,
    )

    # Evaluate
    print(f"\nEvaluating for {args.episodes} episodes...")
    results = evaluate_policy(policy, env, num_episodes=args.episodes)

    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    print(f"Average Episode Length: {results['avg_episode_length']:.1f}")
    print(f"Average Episode Reward: {results['avg_episode_reward']:.2f} ± {results['std_episode_reward']:.2f}")
    print("="*50)

    env.close()


if __name__ == "__main__":
    main()
