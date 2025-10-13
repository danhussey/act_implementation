"""Dataset for loading demonstration trajectories."""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class DemonstrationDataset(Dataset):
    """
    Dataset for ACT training from demonstration trajectories.

    Loads demonstrations from HDF5 file and provides (observation, action_chunk) pairs.
    """

    def __init__(
        self,
        data_path: str,
        chunk_size: int = 10,
        camera_names: Optional[List[str]] = None,
        normalize_actions: bool = True,
        normalize_states: bool = True,
    ):
        """
        Initialize demonstration dataset.

        Args:
            data_path: Path to HDF5 file with demonstrations
            chunk_size: Number of actions in each chunk
            camera_names: List of camera names to load
            normalize_actions: Whether to normalize actions
            normalize_states: Whether to normalize states
        """
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
        self.normalize_actions = normalize_actions
        self.normalize_states = normalize_states

        # Load all episodes into memory
        self.episodes = []
        self._load_episodes()

        # Compute normalization statistics
        if normalize_actions or normalize_states:
            self._compute_normalization_stats()

        # Build index of valid timesteps (where we can extract full chunks)
        self.valid_indices = []
        self._build_index()

    def _load_episodes(self):
        """Load all episodes from HDF5 file."""
        with h5py.File(self.data_path, "r") as f:
            num_episodes = f.attrs["num_episodes"]

            for i in range(num_episodes):
                episode_key = f"episode_{i}"
                episode_data = {}

                # Load states and actions
                episode_data["states"] = f[episode_key]["states"][:]
                episode_data["actions"] = f[episode_key]["actions"][:]

                # Load camera images
                episode_data["images"] = {}
                for cam_name in self.camera_names:
                    key = f"images_{cam_name}"
                    if key in f[episode_key]:
                        episode_data["images"][cam_name] = f[episode_key][key][:]

                self.episodes.append(episode_data)

    def _compute_normalization_stats(self):
        """Compute mean/std for normalization."""
        if self.normalize_states:
            all_states = np.concatenate([ep["states"] for ep in self.episodes], axis=0)
            self.state_mean = torch.from_numpy(all_states.mean(axis=0)).float()
            self.state_std = torch.from_numpy(all_states.std(axis=0) + 1e-6).float()
        else:
            self.state_mean = 0.0
            self.state_std = 1.0

        if self.normalize_actions:
            all_actions = np.concatenate([ep["actions"] for ep in self.episodes], axis=0)
            self.action_mean = torch.from_numpy(all_actions.mean(axis=0)).float()
            self.action_std = torch.from_numpy(all_actions.std(axis=0) + 1e-6).float()
        else:
            self.action_mean = 0.0
            self.action_std = 1.0

    def _build_index(self):
        """Build index of valid (episode_idx, timestep) pairs."""
        for ep_idx, episode in enumerate(self.episodes):
            episode_length = len(episode["states"])

            # For each timestep, we need chunk_size future actions
            for t in range(episode_length):
                # Check if we can extract a full chunk starting from t
                if t + self.chunk_size <= len(episode["actions"]):
                    self.valid_indices.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Returns dict with:
            - state: Robot state (state_dim,)
            - images: Dict of camera images {cam_name: (C, H, W)}
            - actions: Action chunk (chunk_size, action_dim)
        """
        ep_idx, t = self.valid_indices[idx]
        episode = self.episodes[ep_idx]

        # Get current state
        state = torch.from_numpy(episode["states"][t]).float()

        # Normalize state
        if self.normalize_states:
            state = (state - self.state_mean) / self.state_std

        # Get camera images at current timestep
        images = {}
        for cam_name in self.camera_names:
            img = episode["images"][cam_name][t]  # (C, H, W)
            images[cam_name] = torch.from_numpy(img).float()

        # Get action chunk
        action_chunk = episode["actions"][t:t + self.chunk_size]
        actions = torch.from_numpy(action_chunk).float()

        # Normalize actions
        if self.normalize_actions:
            actions = (actions - self.action_mean) / self.action_std

        return {
            "state": state,
            "images": images,
            "actions": actions,
        }

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            "num_episodes": len(self.episodes),
            "num_samples": len(self.valid_indices),
            "state_dim": self.episodes[0]["states"].shape[1],
            "action_dim": self.episodes[0]["actions"].shape[1],
            "state_mean": self.state_mean if self.normalize_states else None,
            "state_std": self.state_std if self.normalize_states else None,
            "action_mean": self.action_mean if self.normalize_actions else None,
            "action_std": self.action_std if self.normalize_actions else None,
        }

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert normalized actions back to original scale."""
        if self.normalize_actions:
            return actions * self.action_std + self.action_mean
        return actions

    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Convert normalized state back to original scale."""
        if self.normalize_states:
            return state * self.state_std + self.state_mean
        return state


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Handles dict of images properly.
    """
    states = torch.stack([item["state"] for item in batch])
    actions = torch.stack([item["actions"] for item in batch])

    # Collate images
    camera_names = list(batch[0]["images"].keys())
    images = {}
    for cam_name in camera_names:
        images[cam_name] = torch.stack([item["images"][cam_name] for item in batch])

    return {
        "state": states,
        "images": images,
        "actions": actions,
    }


def create_dataloaders(
    data_path: str,
    chunk_size: int,
    batch_size: int = 32,
    train_split: float = 0.9,
    num_workers: int = 4,
    camera_names: Optional[List[str]] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_path: Path to HDF5 dataset
        chunk_size: Action chunk size
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of dataloader workers
        camera_names: List of camera names

    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    dataset = DemonstrationDataset(
        data_path=data_path,
        chunk_size=chunk_size,
        camera_names=camera_names,
        normalize_actions=True,
        normalize_states=True,
    )

    # Split into train/val
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
