"""Small ACT training script for robomimic HDF5 demonstrations.

The file is intentionally self-contained so the full training path can be read
top to bottom:

1. Download a small public robomimic dataset.
2. Read action chunks from HDF5.
3. Train a compact ACT-style model to predict future action chunks.

The default path uses robomimic low-dimensional observations, but the script
also supports an image mode that feeds camera frames through a small CNN or an
optional ResNet-18 encoder. Both paths preserve the core ACT idea: predict
several future robot actions at once instead of one action at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DATASETS = {
    "lift-ph": {
        "url": "https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5/lift/ph/low_dim_v15.hdf5?download=true",
        "path": "data/lift_ph_low_dim.hdf5",
    },
    "lift-ph-image": {
        "url": "https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5/lift/ph/demo_v15.hdf5?download=true",
        "path": "data/lift_ph_image.hdf5",
    },
    "can-ph": {
        "url": "https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5/can/ph/low_dim_v15.hdf5?download=true",
        "path": "data/can_ph_low_dim.hdf5",
    },
    "can-ph-image": {
        "url": "https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5/can/ph/demo_v15.hdf5?download=true",
        "path": "data/can_ph_image.hdf5",
    },
}


# Expected robomimic low-dim HDF5 shape:
#
# data/
#   demo_0/
#     actions              (T, action_dim), usually (T, 7)
#     obs/
#       robot0_eef_pos     (T, 3)
#       robot0_eef_quat    (T, 4)
#       object             (T, object_dim)
#       ...
#
# This script trains on low-dimensional observations only. Every 2D dataset in
# obs/ is concatenated into one state vector for each timestep. For example,
# robot0_eef_pos (3) + robot0_eef_quat (4) + robot0_gripper_qpos (2)
# becomes state_dim = 9 before adding any object features.


def download_dataset(name: str, out: Path | None, force: bool) -> Path:
    """Download one public robomimic low-dimensional dataset.

    The downloaded file is already an HDF5 file. No conversion step is needed.
    ``lift-ph`` is smaller and faster for local tests; ``can-ph`` is closer to
    the common pick-and-place task but is larger.

    This helper keeps the project reproducible: someone can clone the repo and
    fetch the same public dataset without hunting for links or manually placing
    files in the expected directory.
    """
    spec = DATASETS[name]
    path = Path(out or spec["path"])
    if path.exists() and not force:
        print(f"exists: {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.unlink(missing_ok=True)
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "act-one-file"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as file:
        total = int(response.headers.get("Content-Length", 0))
        progress = tqdm(total=total or None, unit="B", unit_scale=True)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
            progress.update(len(chunk))
        progress.close()
    tmp.replace(path)
    print(path)
    return path


def sorted_demo_keys(keys) -> list[str]:
    """Sort demo names numerically instead of lexicographically.

    HDF5 keys arrive as strings, so normal sorting would put ``demo_10`` before
    ``demo_2``. This helper keeps the temporal dataset order intuitive.

    It is not required for correctness, but it makes debugging and printed
    subsets easier to reason about because demos appear in their natural order.
    """

    def key(name: str) -> tuple[int, str]:
        suffix = name.removeprefix("demo_").removeprefix("episode_")
        return (int(suffix), name) if suffix.isdigit() else (10**9, name)

    return sorted(keys, key=key)


def decode_value(value) -> str:
    """Decode an HDF5 string-like value into a Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def lowdim_obs_keys(obs: h5py.Group) -> list[str]:
    """Return the ordered robomimic low-dimensional observation keys."""
    keys = []
    for key in sorted(obs.keys()):
        value = obs[key]
        if not isinstance(value, h5py.Dataset) or value.ndim != 2:
            continue
        if key.endswith("_image") or key.endswith("_depth"):
            continue
        keys.append(key)
    if not keys:
        raise ValueError("No low-dimensional observations found in robomimic obs group")
    return keys


def proprio_obs_keys(obs: h5py.Group) -> list[str]:
    """Return robot proprioception keys, excluding privileged object state.

    Vision experiments should make the policy infer object pose from pixels
    instead of reading the robomimic ``object`` vector directly. Keeping robot
    state is still standard: joint/gripper/eef proprioception is available on
    a real robot, while object pose is not.
    """

    keys = [key for key in lowdim_obs_keys(obs) if key.startswith("robot") and not key.endswith("-state")]
    if not keys:
        keys = [key for key in lowdim_obs_keys(obs) if "object" not in key]
    if not keys:
        raise ValueError("No proprioceptive observations found for image mode")
    return keys


def state_from_robomimic_obs(obs: h5py.Group, t: int, keys: list[str] | None = None) -> np.ndarray:
    """Build one flat state vector from all low-dimensional obs arrays at time t.

    robomimic stores observations as named arrays rather than one prebuilt
    vector. Flattening and concatenating them here gives the policy a single
    fixed-size input while preserving all available low-dimensional state.
    """
    parts = [obs[key][t].reshape(-1) for key in (keys or lowdim_obs_keys(obs))]
    return np.concatenate(parts).astype(np.float32)


def image_shape_from_dataset(dataset: h5py.Dataset) -> tuple[int, int, int]:
    """Return image shape as CHW from a robomimic image dataset."""
    if dataset.ndim != 4:
        raise ValueError(f"Expected image dataset with shape (T,H,W,C) or (T,C,H,W), got {dataset.shape}")
    sample_shape = tuple(int(x) for x in dataset.shape[1:])
    if sample_shape[-1] in (1, 3, 4):
        height, width, channels = sample_shape
        return (min(channels, 3), height, width)
    if sample_shape[0] in (1, 3, 4):
        channels, height, width = sample_shape
        return (min(channels, 3), height, width)
    raise ValueError(f"Could not infer channel dimension from image shape {sample_shape}")


def image_to_tensor(image: np.ndarray, target_shape: tuple[int, int, int] | None = None) -> torch.Tensor:
    """Convert one image to a normalized CHW float tensor, resizing if needed."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected one image with 3 dimensions, got shape {array.shape}")
    if array.shape[-1] in (1, 3, 4):
        array = array[..., :3].transpose(2, 0, 1)
    elif array.shape[0] in (1, 3, 4):
        array = array[:3]
    else:
        raise ValueError(f"Could not infer channel dimension from image shape {array.shape}")

    tensor = torch.tensor(np.ascontiguousarray(array), dtype=torch.float32)
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    if target_shape is not None and tuple(tensor.shape) != tuple(target_shape):
        tensor = nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=target_shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


def state_from_env_obs(obs: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    """Build the same flat state vector from a live robosuite observation dict."""
    parts = []
    for key in keys:
        source_key = key
        if source_key not in obs and source_key == "object" and "object-state" in obs:
            source_key = "object-state"
        if source_key not in obs:
            raise KeyError(f"Environment observation is missing expected key '{key}'")
        parts.append(np.asarray(obs[source_key]).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def image_from_env_obs(obs: dict[str, np.ndarray], image_key: str, camera: str, target_shape: tuple[int, int, int]) -> torch.Tensor:
    """Build the model image tensor from a live robosuite observation."""
    key = image_key if image_key in obs else f"{camera}_image"
    if key not in obs:
        raise KeyError(f"Environment observation is missing image key '{image_key}'")
    # robosuite camera frames are vertically flipped relative to normal display.
    return image_to_tensor(obs[key][::-1], target_shape)


class FFmpegVideoWriter:
    """Write rollout frames to mp4 through the local ffmpeg binary."""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        """Launch ffmpeg and prepare to stream raw RGB frames."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for mp4 output but was not found on PATH")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(self.path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        """Write one HWC RGB frame."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Video writer has not been started")
        rgb = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected frame shape {(self.height, self.width, 3)}, got {rgb.shape}")
        self.process.stdin.write(rgb.tobytes())

    def close(self) -> None:
        """Close ffmpeg and raise on encoding failures."""
        if self.process is None:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
            self.process.stderr.close()
        code = self.process.wait()
        self.process = None
        if code != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace"))


class DemoDataset(Dataset):
    """Lazy loader for robomimic HDF5 action chunks.

    ACT trains on overlapping windows. If a demo has ``T=100`` actions and
    ``chunk_size=8``, this dataset exposes 93 samples:

    - sample 0 predicts actions ``0..7`` from observation 0
    - sample 1 predicts actions ``1..8`` from observation 1
    - ...

    The HDF5 file is opened lazily in ``__getitem__`` so constructing the
    dataset does not load all demonstrations into memory.
    """

    def __init__(
        self,
        path: Path,
        chunk_size: int,
        demos: list[str] | None = None,
        stats: dict | None = None,
        max_demos: int | None = None,
        obs_mode: str = "low_dim",
        image_key: str = "agentview_image",
        proprio_keys_: list[str] | None = None,
    ):
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.obs_mode = obs_mode
        self.image_key = image_key
        self.image_shape: tuple[int, int, int] | None = None
        self.obs_keys: list[str] = []
        self.file: h5py.File | None = None
        self.indices: list[tuple[str, int]] = []

        with h5py.File(self.path, "r") as f:
            self.format = "robomimic" if "data" in f else "simple"
            if self.format == "robomimic":
                all_demos = sorted_demo_keys(f["data"].keys())
                self.demos = all_demos[:max_demos] if demos is None and max_demos else (all_demos if demos is None else demos)
                first = f["data"][self.demos[0]]
                if obs_mode == "image":
                    if image_key not in first["obs"]:
                        raise KeyError(f"Image key '{image_key}' was not found in {self.path}")
                    self.obs_keys = proprio_keys_ or proprio_obs_keys(first["obs"])
                    self.image_shape = image_shape_from_dataset(first["obs"][image_key])
                elif obs_mode == "low_dim":
                    self.obs_keys = lowdim_obs_keys(first["obs"])
                else:
                    raise ValueError(f"Unknown obs_mode '{obs_mode}'")
                self.state_dim = state_from_robomimic_obs(first["obs"], 0, self.obs_keys).shape[0]
                self.action_dim = first["actions"].shape[1]
                for demo in self.demos:
                    length = f["data"][demo]["actions"].shape[0]
                    self.indices.extend((demo, t) for t in range(max(0, length - chunk_size + 1)))
            else:
                if obs_mode != "low_dim":
                    raise ValueError("image obs_mode requires a robomimic-style HDF5 file with data/*/obs")
                all_demos = [f"episode_{i}" for i in range(int(f.attrs["num_episodes"]))]
                self.demos = all_demos[:max_demos] if demos is None and max_demos else (all_demos if demos is None else demos)
                first = f[self.demos[0]]
                self.state_dim = first["states"].shape[1]
                self.action_dim = first["actions"].shape[1]
                for demo in self.demos:
                    length = f[demo]["actions"].shape[0]
                    self.indices.extend((demo, t) for t in range(max(0, length - chunk_size + 1)))

        if stats is None:
            self.stats = self._compute_stats()
        else:
            self.stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in stats.items()}

    def _compute_stats(self) -> dict[str, torch.Tensor]:
        """Compute normalization statistics over the selected demonstrations.

        Behavior cloning is much easier when state and action dimensions are on
        comparable scales. The dataset stores means and standard deviations for
        states and actions, then ``__getitem__`` returns normalized tensors.
        """

        states, actions = [], []
        with h5py.File(self.path, "r") as f:
            for demo in self.demos:
                group = f["data"][demo] if self.format == "robomimic" else f[demo]
                if self.format == "robomimic":
                    obs = group["obs"]
                    states.append(np.stack([state_from_robomimic_obs(obs, t, self.obs_keys) for t in range(group["actions"].shape[0])]))
                else:
                    states.append(group["states"][:])
                actions.append(group["actions"][:])
        states_np = np.concatenate(states)
        actions_np = np.concatenate(actions)
        return {
            "state_mean": torch.tensor(states_np.mean(0), dtype=torch.float32),
            "state_std": torch.tensor(states_np.std(0) + 1e-6, dtype=torch.float32),
            "action_mean": torch.tensor(actions_np.mean(0), dtype=torch.float32),
            "action_std": torch.tensor(actions_np.std(0) + 1e-6, dtype=torch.float32),
        }

    def _h5(self) -> h5py.File:
        """Return a persistent HDF5 handle for this Dataset instance.

        Opening an HDF5 file for every sample is slow. Keeping one handle per
        Dataset instance is the simplest lazy-loading pattern for this script.
        """

        if self.file is None:
            self.file = h5py.File(self.path, "r")
        return self.file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        demo, t = self.indices[idx]
        root = self._h5()
        group = root["data"][demo] if self.format == "robomimic" else root[demo]
        if self.format == "robomimic":
            state_np = state_from_robomimic_obs(group["obs"], t, self.obs_keys)
        else:
            state_np = group["states"][t]
        state = torch.tensor(state_np, dtype=torch.float32)
        actions = torch.tensor(group["actions"][t : t + self.chunk_size], dtype=torch.float32)
        # One training item:
        #   state   -> (state_dim,)
        #   actions -> (chunk_size, action_dim)
        #
        # If chunk_size=8 and action_dim=7, actions is (8, 7): the next
        # eight robot commands starting at timestep t.
        item = {
            "state": (state - self.stats["state_mean"]) / self.stats["state_std"],
            "actions": (actions - self.stats["action_mean"]) / self.stats["action_std"],
        }
        if self.obs_mode == "image":
            item["image"] = image_to_tensor(group["obs"][self.image_key][t], self.image_shape)
        return item

    def checkpoint_metadata(self) -> dict:
        """Return dataset metadata needed to rebuild the policy at rollout time."""
        return {
            "obs_mode": self.obs_mode,
            "obs_keys": self.obs_keys,
            "image_key": self.image_key,
            "image_shape": list(self.image_shape) if self.image_shape is not None else None,
        }


def collate(batch: list[dict]) -> dict:
    """Merge individual dataset samples into a training batch.

    PyTorch's default collation would also work for this small dict, but keeping
    this explicit makes the two tensors the model sees very clear.

    This is deliberately boring glue code: it is the boundary where a list of
    Python samples becomes the batched tensor API used by the model.
    """

    # Batch shapes:
    #   state   -> (batch, state_dim)
    #   actions -> (batch, chunk_size, action_dim)
    return {
        "state": torch.stack([x["state"] for x in batch]),
        "actions": torch.stack([x["actions"] for x in batch]),
        **({"image": torch.stack([x["image"] for x in batch])} if "image" in batch[0] else {}),
    }


def split_loaders(
    path: Path,
    chunk_size: int,
    batch_size: int,
    seed: int,
    max_demos: int | None,
    obs_mode: str = "low_dim",
    image_key: str = "agentview_image",
) -> tuple[DemoDataset, DataLoader, DataLoader]:
    """Create train/validation loaders split by whole demonstrations.

    The split is done by demo, not by timestep. This avoids training on one
    chunk from a trajectory and validating on a nearly identical neighboring
    chunk from the same trajectory.
    """

    full = DemoDataset(path, chunk_size, max_demos=max_demos, obs_mode=obs_mode, image_key=image_key)
    rng = np.random.default_rng(seed)
    demos = np.array(full.demos)
    rng.shuffle(demos)
    cut = max(1, int(0.9 * len(demos)))
    train_demos = demos[:cut].tolist()
    val_demos = demos[cut:].tolist() or train_demos
    stats = {k: v.tolist() for k, v in full.stats.items()}
    train_set = DemoDataset(path, chunk_size, train_demos, stats=stats, obs_mode=obs_mode, image_key=image_key, proprio_keys_=full.obs_keys)
    val_set = DemoDataset(path, chunk_size, val_demos, stats=stats, obs_mode=obs_mode, image_key=image_key, proprio_keys_=full.obs_keys)
    return full, DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate), DataLoader(val_set, batch_size, collate_fn=collate)


class PositionalEncoding(nn.Module):
    """Sinusoidal position signal for action-query tokens.

    The transformer decoder receives ``chunk_size`` learned query vectors. The
    positional encoding tells the model which query means "first future action",
    "second future action", etc.
    """

    def __init__(self, dim: int, length: int):
        super().__init__()
        pos = torch.arange(length).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, 1, dim)
        pe[:, 0, 0::2] = torch.sin(pos * div)
        pe[:, 0, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


class ScratchCNNEncoder(nn.Module):
    """Small train-from-scratch image encoder for laptop-scale experiments."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(128, out_dim), nn.ReLU())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(image))


class ResNet18Encoder(nn.Module):
    """Optional torchvision ResNet-18 encoder, with optional pretrained weights."""

    def __init__(self, out_dim: int, pretrained: bool, freeze: bool, load_weights: bool = True):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError("Install torchvision to use --vision-backbone resnet18") from exc

        weights = ResNet18_Weights.DEFAULT if pretrained and load_weights else None
        base = resnet18(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(base.fc.in_features, out_dim), nn.ReLU())
        self.pretrained = pretrained
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            image = (image - self.mean) / self.std
        return self.proj(self.features(image))


def make_vision_encoder(backbone: str, out_dim: int, pretrained: bool, freeze: bool, load_pretrained_weights: bool = True) -> nn.Module:
    """Build the configured image encoder."""
    if backbone == "scratch_cnn":
        if pretrained:
            raise ValueError("scratch_cnn does not support --vision-pretrained")
        encoder = ScratchCNNEncoder(out_dim)
        if freeze:
            for param in encoder.features.parameters():
                param.requires_grad = False
        return encoder
    if backbone == "resnet18":
        return ResNet18Encoder(out_dim, pretrained=pretrained, freeze=freeze, load_weights=load_pretrained_weights)
    raise ValueError(f"Unknown vision backbone '{backbone}'")


class ACT(nn.Module):
    """ACT-style chunk predictor.

    Input:
      state:   (batch, state_dim)
      actions: (batch, chunk_size, action_dim) during training, None at inference

    Output:
      predicted actions: (batch, chunk_size, action_dim)

    This is a compact version of ACT:
    - ``obs`` embeds the current robot/object state.
    - ``latent`` encodes the ground-truth action chunk during training, giving
      the model a CVAE-style latent variable for multimodal demonstrations.
    - ``query`` contains one learned token per future action.
    - the transformer decoder turns those query tokens into an action chunk.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        dim: int = 128,
        latent_dim: int = 16,
        image_shape: tuple[int, int, int] | None = None,
        vision_backbone: str = "scratch_cnn",
        vision_pretrained: bool = False,
        freeze_vision: bool = False,
        vision_load_pretrained_weights: bool = True,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.obs = nn.Linear(state_dim, dim)
        self.image_encoder = None
        self.fuse = None
        if image_shape is not None:
            if image_shape[0] != 3:
                raise ValueError(f"Expected 3-channel images, got image shape {image_shape}")
            self.image_encoder = make_vision_encoder(vision_backbone, dim, vision_pretrained, freeze_vision, vision_load_pretrained_weights)
            self.fuse = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.latent = nn.Sequential(nn.Linear(chunk_size * action_dim + dim, dim), nn.ReLU(), nn.Linear(dim, latent_dim * 2))
        self.z_proj = nn.Linear(latent_dim, dim)
        self.query = nn.Parameter(torch.randn(chunk_size, 1, dim))
        self.pos = PositionalEncoding(dim, chunk_size)
        enc = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True)
        dec = nn.TransformerDecoderLayer(dim, 4, dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, 2)
        self.decoder = nn.TransformerDecoder(dec, 2)
        self.head = nn.Linear(dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict a chunk of future actions.

        During training, ``actions`` is provided so the latent encoder can learn
        ``q(z | actions, state)``. During inference, ``actions`` is ``None`` and
        callers can pass an explicit latent or fall back to ``z=0``.
        """

        batch = state.size(0)
        obs = self.obs(state)
        if self.image_encoder is not None:
            if image is None:
                raise ValueError("This checkpoint expects image observations")
            image_obs = self.image_encoder(image)
            obs = self.fuse(torch.cat([obs, image_obs], dim=1))
        memory = self.encoder(obs.unsqueeze(1))
        if actions is None:
            mu = logvar = torch.zeros(batch, self.latent_dim, device=state.device)
            if z is None:
                z = mu
            elif z.dim() == 1:
                z = z.unsqueeze(0).expand(batch, -1)
            if z.shape != (batch, self.latent_dim):
                raise ValueError(f"Expected latent shape {(batch, self.latent_dim)}, got {tuple(z.shape)}")
        else:
            latent = self.latent(torch.cat([actions.flatten(1), obs], dim=1))
            mu, logvar = latent.chunk(2, dim=1)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        query = self.pos(self.query.expand(-1, batch, -1)).transpose(0, 1)
        query = query + self.z_proj(z).unsqueeze(1)
        return self.head(self.decoder(query, memory)), mu, logvar


def loss_fn(pred: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:
    """ACT training objective: reconstruction plus KL regularization.

    ``recon`` teaches the predicted chunk to match demonstration actions.
    ``kl`` keeps the learned latent distribution close to a unit Gaussian so
    inference can use a simple zero/prior latent.
    """

    recon = nn.functional.mse_loss(pred, target)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon + beta * kl


def run_epoch(model: ACT, loader: DataLoader, device: torch.device, optimizer=None, beta: float = 10.0) -> float:
    """Run one training or validation epoch.

    Passing an optimizer enables gradient updates. Passing ``optimizer=None``
    switches to evaluation mode and computes loss without backpropagation.
    """

    model.train(optimizer is not None)
    total = 0.0
    for batch in tqdm(loader, leave=False):
        states = batch["state"].to(device)
        actions = batch["actions"].to(device)
        images = batch.get("image")
        if images is not None:
            images = images.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            pred, mu, logvar = model(states, actions, image=images)
            loss = loss_fn(pred, actions, mu, logvar, beta)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def checkpoint_payload(model: ACT, dataset: DemoDataset, args: argparse.Namespace) -> dict:
    """Build the checkpoint dict used by training, rollout, and resume."""
    return {
        "model": model.state_dict(),
        "stats": {k: v.tolist() for k, v in dataset.stats.items()},
        "args": vars(args),
        "dataset": dataset.checkpoint_metadata(),
    }


def stats_to_device(stats: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move normalization stats to the rollout device."""
    return {key: value.to(device) for key, value in stats.items()}


def assert_resume_compatible(checkpoint: dict, dataset: DemoDataset, args: argparse.Namespace) -> None:
    """Fail early if a resume checkpoint does not match this training setup."""
    train_args = checkpoint.get("args", {})
    dataset_meta = checkpoint.get("dataset", {})
    checks = {
        "chunk_size": args.chunk_size,
        "dim": args.dim,
        "obs_mode": args.obs_mode,
        "image_key": args.image_key,
        "vision_backbone": args.vision_backbone,
        "vision_pretrained": args.vision_pretrained,
        "freeze_vision": args.freeze_vision,
    }
    mismatches = [key for key, value in checks.items() if train_args.get(key, value) != value]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(f"Resume checkpoint is incompatible with current args: {joined}")

    if dataset_meta.get("obs_keys") and dataset_meta["obs_keys"] != dataset.obs_keys:
        raise ValueError("Resume checkpoint observation keys do not match the current dataset")
    for key, value in checkpoint["stats"].items():
        expected = dataset.stats[key]
        actual = torch.tensor(value, dtype=torch.float32)
        if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
            raise ValueError(f"Resume checkpoint normalization stat '{key}' differs from the current dataset")


def train(args: argparse.Namespace) -> None:
    """Train ACT and save the best checkpoint plus a tiny metrics file.

    Keeping orchestration in one function makes the end-to-end flow obvious:
    build loaders, build model, optimize, checkpoint the best validation loss.
    """

    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full, train_loader, val_loader = split_loaders(
        Path(args.data),
        args.chunk_size,
        args.batch_size,
        args.seed,
        args.max_demos,
        obs_mode=args.obs_mode,
        image_key=args.image_key,
    )
    model = ACT(
        full.state_dim,
        full.action_dim,
        args.chunk_size,
        args.dim,
        image_shape=full.image_shape,
        vision_backbone=args.vision_backbone,
        vision_pretrained=args.vision_pretrained,
        freeze_vision=args.freeze_vision,
    ).to(device)
    if args.resume:
        checkpoint = torch.load(Path(args.resume), map_location=device)
        assert_resume_compatible(checkpoint, full, args)
        model.load_state_dict(checkpoint["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    best_epoch = 0
    started_at = time.perf_counter()
    max_seconds = args.max_minutes * 60 if args.max_minutes is not None else None
    history_path = out / "history.jsonl"
    rollout_history_path = out / "rollout_history.jsonl"
    history_path.unlink(missing_ok=True)
    rollout_history_path.unlink(missing_ok=True)

    def maybe_eval_rollouts(epoch: int, elapsed_seconds: float, tag: str) -> dict | None:
        if args.eval_episodes <= 0:
            return None
        video_dir = out / "rollout_videos" / tag if args.eval_videos else None
        summary = evaluate_model_rollouts(
            model=model,
            stats=stats_to_device(full.stats, device),
            obs_keys=full.obs_keys,
            obs_mode=full.obs_mode,
            image_key=full.image_key,
            image_shape=full.image_shape,
            data_path=Path(args.data),
            device=device,
            episodes=args.eval_episodes,
            start_demo_index=args.eval_start_demo_index,
            camera=args.eval_camera,
            width=args.eval_width,
            height=args.eval_height,
            fps=args.eval_fps,
            max_steps=args.eval_max_steps,
            videos=args.eval_videos,
            video_dir=video_dir,
            desc=f"Rollout eval {tag}",
        )
        rollout_row = {
            "epoch": epoch,
            "tag": tag,
            "elapsed_seconds": elapsed_seconds,
            **summary,
        }
        with rollout_history_path.open("a") as f:
            f.write(json.dumps(rollout_row) + "\n")
        return rollout_row

    if args.eval_before_train:
        rollout_row = maybe_eval_rollouts(0, 0.0, "epoch_0000")
        if rollout_row is not None:
            print(
                f"rollout epoch=0 success={rollout_row['successes']}/{rollout_row['episodes']} "
                f"rate={rollout_row['success_rate']:.2f}"
            )

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        elapsed_seconds = time.perf_counter() - started_at
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_seconds": elapsed_seconds,
        }
        should_eval = args.eval_every_epochs > 0 and (epoch + 1) % args.eval_every_epochs == 0
        if should_eval:
            rollout_row = maybe_eval_rollouts(epoch + 1, elapsed_seconds, f"epoch_{epoch + 1:04d}")
            if rollout_row is not None:
                row.update(
                    {
                        "rollout_successes": rollout_row["successes"],
                        "rollout_episodes": rollout_row["episodes"],
                        "rollout_success_rate": rollout_row["success_rate"],
                        "rollout_avg_reward": rollout_row["avg_reward"],
                        "rollout_avg_steps": rollout_row["avg_steps"],
                    }
                )
                elapsed_seconds = time.perf_counter() - started_at
                row["elapsed_seconds"] = elapsed_seconds
        with history_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        rollout_part = ""
        if "rollout_success_rate" in row:
            rollout_part = f" rollout={row['rollout_successes']}/{row['rollout_episodes']} ({row['rollout_success_rate']:.2f})"
        print(f"epoch={epoch + 1} train={train_loss:.4f} val={val_loss:.4f}{rollout_part} elapsed={elapsed_seconds / 60:.1f}m")
        if val_loss < best:
            best = val_loss
            best_epoch = epoch + 1
            torch.save(checkpoint_payload(model, full, args), out / "best.pt")
        if max_seconds is not None and elapsed_seconds >= max_seconds:
            break
    torch.save(checkpoint_payload(model, full, args), out / "last.pt")
    metrics = {
        "best_val_loss": best,
        "best_epoch": best_epoch,
        "epochs_completed": epoch + 1,
        "elapsed_seconds": time.perf_counter() - started_at,
        "demos": len(full.demos),
        "resume": args.resume,
        "obs_mode": args.obs_mode,
        "image_key": args.image_key if args.obs_mode == "image" else None,
        "vision_backbone": args.vision_backbone if args.obs_mode == "image" else None,
        "vision_pretrained": args.vision_pretrained if args.obs_mode == "image" else None,
        "rollout_history": str(rollout_history_path) if rollout_history_path.exists() else None,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_jsonl(path: Path) -> list[dict]:
    """Load one non-empty JSONL file."""
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_history(path: Path) -> list[dict]:
    """Load one JSONL training history file."""
    rows = load_jsonl(path)
    return rows


def make_history_summary(rows: list[dict]) -> dict:
    """Return compact training observability metrics from history rows."""
    best = min(rows, key=lambda row: row["val_loss"])
    last = rows[-1]
    return {
        "epochs": len(rows),
        "elapsed_seconds": last["elapsed_seconds"],
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "last_epoch": last["epoch"],
        "last_train_loss": last["train_loss"],
        "last_val_loss": last["val_loss"],
        "epochs_per_minute": len(rows) / max(1e-9, last["elapsed_seconds"] / 60),
    }


def make_rollout_summary(rows: list[dict]) -> dict:
    """Return compact closed-loop observability metrics from rollout probes."""
    best = max(rows, key=lambda row: row["success_rate"])
    last = rows[-1]
    return {
        "rollout_evals": len(rows),
        "best_rollout_epoch": best["epoch"],
        "best_rollout_successes": best["successes"],
        "best_rollout_episodes": best["episodes"],
        "best_rollout_success_rate": best["success_rate"],
        "last_rollout_epoch": last["epoch"],
        "last_rollout_successes": last["successes"],
        "last_rollout_episodes": last["episodes"],
        "last_rollout_success_rate": last["success_rate"],
    }


def write_loss_curve_svg(rows: list[dict], out: Path, title: str) -> None:
    """Write a small self-contained SVG loss chart without plotting deps."""
    width, height = 920, 520
    left, right, top, bottom = 76, 28, 52, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    epochs = [row["epoch"] for row in rows]
    train = [row["train_loss"] for row in rows]
    val = [row["val_loss"] for row in rows]
    best_so_far = []
    running = float("inf")
    for value in val:
        running = min(running, value)
        best_so_far.append(running)

    x_min, x_max = min(epochs), max(epochs)
    y_values = train + val
    y_min, y_max = min(y_values), max(y_values)
    pad = (y_max - y_min) * 0.08 or 0.1
    y_min = max(0.0, y_min - pad)
    y_max += pad

    def x_scale(epoch: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (epoch - x_min) / (x_max - x_min) * plot_w

    def y_scale(loss: float) -> float:
        return top + (y_max - loss) / (y_max - y_min) * plot_h

    def points(values: list[float]) -> str:
        return " ".join(f"{x_scale(epoch):.1f},{y_scale(value):.1f}" for epoch, value in zip(epochs, values))

    y_ticks = [y_min + index * (y_max - y_min) / 5 for index in range(6)]
    x_ticks = [round(x_min + index * (x_max - x_min) / 5) for index in range(6)]
    best_index, best_value = min(enumerate(val), key=lambda item: item[1])
    best_epoch = epochs[best_index]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#1f2937} .axis{stroke:#374151;stroke-width:1.4} .grid{stroke:#e5e7eb;stroke-width:1} .train{fill:none;stroke:#2563eb;stroke-width:2.2} .val{fill:none;stroke:#dc2626;stroke-width:2.2} .best{fill:none;stroke:#059669;stroke-width:2;stroke-dasharray:5 5}</style>",
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">{title}</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
    ]
    for tick in y_ticks:
        y = y_scale(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{tick:.3f}</text>')
    for tick in x_ticks:
        x = x_scale(tick)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" font-size="11" text-anchor="middle">{tick}</text>')
    parts.extend(
        [
            f'<polyline class="train" points="{points(train)}"/>',
            f'<polyline class="val" points="{points(val)}"/>',
            f'<polyline class="best" points="{points(best_so_far)}"/>',
            f'<circle cx="{x_scale(best_epoch):.1f}" cy="{y_scale(best_value):.1f}" r="4.5" fill="#059669"/>',
            f'<text x="{left + 4}" y="{height - 26}" font-size="12">epoch</text>',
            f'<text x="18" y="{top + 18}" font-size="12" transform="rotate(-90 18,{top + 18})">loss</text>',
            f'<rect x="{left + plot_w - 222}" y="{top + 14}" width="206" height="72" rx="8" fill="#ffffff" stroke="#e5e7eb"/>',
            f'<line x1="{left + plot_w - 204}" y1="{top + 36}" x2="{left + plot_w - 164}" y2="{top + 36}" class="train"/><text x="{left + plot_w - 154}" y="{top + 40}" font-size="12">train</text>',
            f'<line x1="{left + plot_w - 204}" y1="{top + 56}" x2="{left + plot_w - 164}" y2="{top + 56}" class="val"/><text x="{left + plot_w - 154}" y="{top + 60}" font-size="12">val</text>',
            f'<line x1="{left + plot_w - 204}" y1="{top + 76}" x2="{left + plot_w - 164}" y2="{top + 76}" class="best"/><text x="{left + plot_w - 154}" y="{top + 80}" font-size="12">best val</text>',
            f'<text x="{left}" y="{height - 46}" font-size="12">best val {best_value:.4f} at epoch {best_epoch} | last train {train[-1]:.4f}, val {val[-1]:.4f}</text>',
            "</svg>",
        ]
    )
    out.write_text("\n".join(parts))


def write_rollout_curve_svg(rows: list[dict], out: Path, title: str) -> None:
    """Write a small self-contained SVG rollout success chart."""
    width, height = 920, 420
    left, right, top, bottom = 76, 30, 52, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    epochs = [row["epoch"] for row in rows]
    rates = [row["success_rate"] for row in rows]
    x_min, x_max = min(epochs), max(epochs)

    def x_scale(epoch: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (epoch - x_min) / (x_max - x_min) * plot_w

    def y_scale(rate: float) -> float:
        return top + (1.0 - rate) * plot_h

    def points() -> str:
        return " ".join(f"{x_scale(epoch):.1f},{y_scale(rate):.1f}" for epoch, rate in zip(epochs, rates))

    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_ticks = [round(x_min + index * (x_max - x_min) / 5) for index in range(6)]
    best = max(rows, key=lambda row: row["success_rate"])
    last = rows[-1]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#1f2937} .axis{stroke:#374151;stroke-width:1.4} .grid{stroke:#e5e7eb;stroke-width:1} .success{fill:none;stroke:#0f766e;stroke-width:2.8} .point{fill:#0f766e;stroke:#fbfaf7;stroke-width:2}</style>",
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="30" font-size="18" font-weight="700">{title} rollout success</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
    ]
    for tick in y_ticks:
        y = y_scale(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{tick * 100:.0f}%</text>')
    for tick in x_ticks:
        x = x_scale(tick)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" font-size="11" text-anchor="middle">{tick}</text>')
    parts.append(f'<polyline class="success" points="{points()}"/>')
    for row in rows:
        x = x_scale(row["epoch"])
        y = y_scale(row["success_rate"])
        parts.append(f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{max(16, y - 12):.1f}" font-size="11" text-anchor="middle">{row["successes"]}/{row["episodes"]}</text>'
        )
    parts.extend(
        [
            f'<text x="{left + 4}" y="{height - 24}" font-size="12">epoch</text>',
            f'<text x="18" y="{top + 18}" font-size="12" transform="rotate(-90 18,{top + 18})">rollout success</text>',
            f'<text x="{left}" y="{height - 44}" font-size="12">best {best["successes"]}/{best["episodes"]} at epoch {best["epoch"]} | last {last["successes"]}/{last["episodes"]} at epoch {last["epoch"]}</text>',
            "</svg>",
        ]
    )
    out.write_text("\n".join(parts))


def plot_history(args: argparse.Namespace) -> None:
    """Generate loss and rollout observability artifacts for a training run."""
    run_dir = Path(args.run)
    history_path = Path(args.history) if args.history else run_dir / "history.jsonl"
    out = Path(args.out) if args.out else run_dir / "loss_curve.svg"
    summary_path = Path(args.summary) if args.summary else run_dir / "history_summary.json"
    rows = load_history(history_path)
    summary = make_history_summary(rows)
    summary_path.write_text(json.dumps(summary, indent=2))
    write_loss_curve_svg(rows, out, args.title or run_dir.name)
    result = {"history": str(history_path), "curve": str(out), "summary": str(summary_path), **summary}

    rollout_history_path = run_dir / "rollout_history.jsonl"
    if rollout_history_path.exists() and rollout_history_path.read_text().strip():
        rollout_rows = load_jsonl(rollout_history_path)
        rollout_curve = run_dir / "rollout_curve.svg"
        rollout_summary_path = run_dir / "rollout_summary.json"
        rollout_summary = make_rollout_summary(rollout_rows)
        rollout_summary_path.write_text(json.dumps(rollout_summary, indent=2))
        write_rollout_curve_svg(rollout_rows, rollout_curve, args.title or run_dir.name)
        result.update(
            {
                "rollout_history": str(rollout_history_path),
                "rollout_curve": str(rollout_curve),
                "rollout_summary": str(rollout_summary_path),
                **rollout_summary,
            }
        )
    print(json.dumps(result, indent=2))


@dataclass
class RolloutAssets:
    """Loaded checkpoint state needed for closed-loop simulator evaluation."""

    model: ACT
    stats: dict[str, torch.Tensor]
    obs_keys: list[str]
    obs_mode: str
    image_key: str
    image_shape: tuple[int, int, int] | None


def load_rollout_assets(checkpoint_path: Path, data_path: Path, device: torch.device) -> RolloutAssets:
    """Load a trained ACT checkpoint plus the observation keys needed for rollout."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint["args"]
    dataset_meta = checkpoint.get("dataset", {})
    obs_mode = dataset_meta.get("obs_mode", train_args.get("obs_mode", "low_dim"))
    image_key = dataset_meta.get("image_key", train_args.get("image_key", "agentview_image"))
    with h5py.File(data_path, "r") as f:
        demos = sorted_demo_keys(f["data"].keys())
        first_obs = f["data"][demos[0]]["obs"]
        if obs_mode == "image":
            obs_keys = dataset_meta.get("obs_keys") or proprio_obs_keys(first_obs)
            image_shape_raw = dataset_meta.get("image_shape")
            if image_shape_raw is None:
                image_shape = image_shape_from_dataset(first_obs[image_key])
            else:
                image_shape = tuple(int(x) for x in image_shape_raw)
        else:
            obs_keys = dataset_meta.get("obs_keys") or lowdim_obs_keys(first_obs)
            image_shape = None
        state_dim = state_from_robomimic_obs(f["data"][demos[0]]["obs"], 0, obs_keys).shape[0]
        action_dim = f["data"][demos[0]]["actions"].shape[1]

    model = ACT(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=train_args["chunk_size"],
        dim=train_args["dim"],
        image_shape=image_shape,
        vision_backbone=train_args.get("vision_backbone", "scratch_cnn"),
        vision_pretrained=train_args.get("vision_pretrained", False),
        freeze_vision=train_args.get("freeze_vision", False),
        vision_load_pretrained_weights=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stats = {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in checkpoint["stats"].items()
    }
    return RolloutAssets(model=model, stats=stats, obs_keys=obs_keys, obs_mode=obs_mode, image_key=image_key, image_shape=image_shape)


def make_robosuite_env(data_path: Path, camera: str, width: int, height: int, horizon: int):
    """Create a robosuite environment from robomimic dataset metadata."""
    import robosuite as suite

    with h5py.File(data_path, "r") as f:
        env_meta = json.loads(decode_value(f["data"].attrs["env_args"]))
    return make_env_from_meta(env_meta, camera, width, height, horizon)


def make_env_from_meta(env_meta: dict, camera: str, width: int, height: int, horizon: int):
    """Create a robosuite environment from decoded robomimic metadata."""
    import robosuite as suite

    kwargs = dict(env_meta["env_kwargs"])
    kwargs.update(
        {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "use_object_obs": False,
            "camera_names": [camera],
            "camera_widths": width,
            "camera_heights": height,
            "horizon": horizon,
        }
    )
    return suite.make(env_name=env_meta["env_name"], **kwargs)


def set_env_state(env, state: np.ndarray) -> dict[str, np.ndarray]:
    """Set robosuite to a specific serialized simulator state."""
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env.update_state()
    return env._get_observations(force_update=True)


def reset_env_to_demo_start(env, data_path: Path, demo_name: str) -> dict[str, np.ndarray]:
    """Reset the robosuite env to the exact initial simulator state of one demo."""
    with h5py.File(data_path, "r") as f:
        demo = f["data"][demo_name]
        model_xml = decode_value(demo.attrs["model_file"])
        initial_state = demo["states"][0]

    env.reset()
    env.reset_from_xml_string(model_xml)
    return set_env_state(env, initial_state)


def render_image_dataset(args: argparse.Namespace) -> None:
    """Render a raw robomimic state/action file into image-observation HDF5."""
    src_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    with h5py.File(src_path, "r") as src:
        env_meta = json.loads(decode_value(src["data"].attrs["env_args"]))
        demos = sorted_demo_keys(src["data"].keys())[: args.max_demos]
        env = make_env_from_meta(env_meta, args.camera, args.width, args.height, args.max_steps)
        try:
            with h5py.File(out_path, "w") as dst:
                data_out = dst.create_group("data")
                data_out.attrs["env_args"] = src["data"].attrs["env_args"]
                total = 0
                kept_obs_keys: list[str] | None = None
                for demo_name in tqdm(demos, desc="Rendering image obs"):
                    demo_in = src["data"][demo_name]
                    states = demo_in["states"][:]
                    actions = demo_in["actions"][:]
                    model_xml = decode_value(demo_in.attrs["model_file"])
                    length = min(len(actions), len(states))
                    demo_out = data_out.create_group(demo_name)
                    demo_out.attrs["model_file"] = demo_in.attrs["model_file"]
                    demo_out.attrs["num_samples"] = length
                    demo_out.create_dataset("actions", data=actions[:length].astype(np.float32))
                    demo_out.create_dataset("states", data=states[:length].astype(np.float32))
                    obs_out = demo_out.create_group("obs")
                    frames = []
                    lowdim_values: dict[str, list[np.ndarray]] = {}
                    env.reset()
                    env.reset_from_xml_string(model_xml)
                    for state in states[:length]:
                        obs = set_env_state(env, state)
                        frame_key = f"{args.camera}_image"
                        if frame_key not in obs:
                            raise KeyError(f"Camera '{args.camera}' not present in rendered observations")
                        frames.append(obs[frame_key][::-1].astype(np.uint8))
                        if kept_obs_keys is None:
                            kept_obs_keys = sorted(
                                key
                                for key, value in obs.items()
                                if key.startswith("robot")
                                and np.asarray(value).ndim <= 1
                                and not key.endswith("_image")
                                and not key.endswith("-state")
                            )
                            if not kept_obs_keys:
                                raise ValueError("Rendered observations did not include robot proprioception keys")
                        for key in kept_obs_keys:
                            lowdim_values.setdefault(key, []).append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
                    obs_out.create_dataset(args.image_key, data=np.stack(frames), compression="gzip", compression_opts=1)
                    for key, values in lowdim_values.items():
                        obs_out.create_dataset(key, data=np.stack(values).astype(np.float32))
                    total += length
                data_out.attrs["total"] = total
        finally:
            env.close()
    print(json.dumps({"source": str(src_path), "out": str(out_path), "demos": len(demos), "total": total}, indent=2))


def run_policy_episode(
    model: ACT,
    stats: dict[str, torch.Tensor],
    obs_keys: list[str],
    obs_mode: str,
    image_key: str,
    image_shape: tuple[int, int, int] | None,
    env,
    data_path: Path,
    demo_name: str,
    camera: str,
    device: torch.device,
    max_steps: int,
    video_path: Path | None = None,
    width: int = 256,
    height: int = 256,
    fps: int = 20,
    latent: torch.Tensor | None = None,
) -> dict:
    """Run one policy attempt, optionally recording it to mp4."""
    obs = reset_env_to_demo_start(env, data_path, demo_name)
    frame_key = f"{camera}_image"
    if frame_key not in obs:
        raise KeyError(f"Camera '{camera}' not present in environment observation")

    writer = None
    if video_path is not None:
        writer = FFmpegVideoWriter(video_path, width, height, fps)
        writer.start()
        writer.write(obs[frame_key][::-1])

    total_reward = 0.0
    success = False
    steps = 0
    first_action = None
    action_norms = []
    try:
        for step in range(max_steps):
            state = torch.tensor(state_from_env_obs(obs, obs_keys), dtype=torch.float32, device=device).unsqueeze(0)
            state = (state - stats["state_mean"]) / stats["state_std"]
            image = None
            if obs_mode == "image":
                if image_shape is None:
                    raise ValueError("image_shape is required for image rollouts")
                image = image_from_env_obs(obs, image_key, camera, image_shape).to(device).unsqueeze(0)
            with torch.no_grad():
                pred, _, _ = model(state, None, z=latent, image=image)
            action = pred[0, 0] * stats["action_std"] + stats["action_mean"]
            action_np = action.detach().cpu().numpy()
            if first_action is None:
                first_action = action_np.tolist()
            action_norms.append(float(np.linalg.norm(action_np)))
            obs, reward, done, info = env.step(action_np)
            if writer is not None:
                writer.write(obs[frame_key][::-1])
            total_reward += reward
            steps = step + 1
            success = bool(info.get("success", False) or getattr(env, "_check_success", lambda: False)())
            if done or success:
                break
    finally:
        if writer is not None:
            writer.close()

    return {
        "demo": demo_name,
        "steps": steps,
        "reward": total_reward,
        "success": success,
        "video": str(video_path) if video_path is not None else None,
        "first_action": first_action,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,
    }


def evaluate_model_rollouts(
    model: ACT,
    stats: dict[str, torch.Tensor],
    obs_keys: list[str],
    obs_mode: str,
    image_key: str,
    image_shape: tuple[int, int, int] | None,
    data_path: Path,
    device: torch.device,
    episodes: int,
    start_demo_index: int,
    camera: str,
    width: int,
    height: int,
    fps: int,
    max_steps: int,
    videos: int = 0,
    video_dir: Path | None = None,
    desc: str = "Evaluating",
) -> dict:
    """Run closed-loop policy attempts from the current in-memory model."""
    if videos and video_dir is None:
        raise ValueError("video_dir is required when videos > 0")
    if video_dir is not None and videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as f:
        demos = sorted_demo_keys(f["data"].keys())

    env = make_robosuite_env(data_path, camera, width, height, max_steps)
    was_training = model.training
    model.eval()
    results = []
    try:
        for episode in tqdm(range(episodes), desc=desc, leave=False):
            demo_index = (start_demo_index + episode) % len(demos)
            demo_name = demos[demo_index]
            video_path = None
            if video_dir is not None and episode < videos:
                video_path = video_dir / f"rollout_{episode:03d}_demo_{demo_index}.mp4"
            result = run_policy_episode(
                model=model,
                stats=stats,
                obs_keys=obs_keys,
                obs_mode=obs_mode,
                image_key=image_key,
                image_shape=image_shape,
                env=env,
                data_path=data_path,
                demo_name=demo_name,
                camera=camera,
                device=device,
                max_steps=max_steps,
                video_path=video_path,
                width=width,
                height=height,
                fps=fps,
            )
            results.append(result)
    finally:
        env.close()
        model.train(was_training)

    successes = sum(int(item["success"]) for item in results)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(1, episodes),
        "avg_reward": float(np.mean([item["reward"] for item in results])) if results else 0.0,
        "avg_steps": float(np.mean([item["steps"] for item in results])) if results else 0.0,
        "results": results,
    }


def rollout(args: argparse.Namespace) -> None:
    """Run a trained ACT policy in robosuite and save an mp4."""
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)

    assets = load_rollout_assets(checkpoint_path, data_path, device)
    with h5py.File(data_path, "r") as f:
        demos = sorted_demo_keys(f["data"].keys())
    demo_name = demos[args.demo_index]

    env = make_robosuite_env(data_path, args.camera, args.width, args.height, args.max_steps)
    try:
        result = run_policy_episode(
            model=assets.model,
            stats=assets.stats,
            obs_keys=assets.obs_keys,
            obs_mode=assets.obs_mode,
            image_key=assets.image_key,
            image_shape=assets.image_shape,
            env=env,
            data_path=data_path,
            demo_name=demo_name,
            camera=args.camera,
            device=device,
            max_steps=args.max_steps,
            video_path=Path(args.out),
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    finally:
        env.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        **result,
    }
    print(json.dumps(summary, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    """Run multiple rollouts and write success-rate metrics plus sample videos."""
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    assets = load_rollout_assets(checkpoint_path, data_path, device)
    with h5py.File(data_path, "r") as f:
        demos = sorted_demo_keys(f["data"].keys())

    env = make_robosuite_env(data_path, args.camera, args.width, args.height, args.max_steps)
    results = []
    try:
        for episode in tqdm(range(args.episodes), desc="Evaluating"):
            demo_index = (args.start_demo_index + episode) % len(demos)
            demo_name = demos[demo_index]
            video_path = None
            if episode < args.videos:
                video_path = videos_dir / f"rollout_{episode:03d}_demo_{demo_index}.mp4"
            result = run_policy_episode(
                model=assets.model,
                stats=assets.stats,
                obs_keys=assets.obs_keys,
                obs_mode=assets.obs_mode,
                image_key=assets.image_key,
                image_shape=assets.image_shape,
                env=env,
                data_path=data_path,
                demo_name=demo_name,
                camera=args.camera,
                device=device,
                max_steps=args.max_steps,
                video_path=video_path,
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
            results.append(result)
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in results)
    summary = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / max(1, args.episodes),
        "avg_reward": float(np.mean([item["reward"] for item in results])) if results else 0.0,
        "avg_steps": float(np.mean([item["steps"] for item in results])) if results else 0.0,
        "results": results,
    }
    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"} | {"metrics": str(metrics_path)}, indent=2))


def latent_sweep(args: argparse.Namespace) -> None:
    """Evaluate fixed latent choices from the same initial state."""
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    assets = load_rollout_assets(checkpoint_path, data_path, device)
    with h5py.File(data_path, "r") as f:
        demos = sorted_demo_keys(f["data"].keys())
    demo_name = demos[args.demo_index]

    latent_specs = []
    if args.include_zero:
        latent_specs.append(("zero", torch.zeros(assets.model.latent_dim), "z=0"))

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for index in range(args.samples):
        latent = torch.randn(assets.model.latent_dim, generator=generator) * args.scale
        latent_specs.append((f"sample_{index:03d}", latent, f"N(0,{args.scale:g}) sample {index}"))

    env = make_robosuite_env(data_path, args.camera, args.width, args.height, args.max_steps)
    results = []
    try:
        for index, (label, latent_cpu, description) in enumerate(tqdm(latent_specs, desc="Latent sweep")):
            video_path = None
            if index < args.videos:
                video_path = videos_dir / f"{index:03d}_{label}.mp4"
            latent = latent_cpu.to(device)
            result = run_policy_episode(
                model=assets.model,
                stats=assets.stats,
                obs_keys=assets.obs_keys,
                obs_mode=assets.obs_mode,
                image_key=assets.image_key,
                image_shape=assets.image_shape,
                env=env,
                data_path=data_path,
                demo_name=demo_name,
                camera=args.camera,
                device=device,
                max_steps=args.max_steps,
                video_path=video_path,
                width=args.width,
                height=args.height,
                fps=args.fps,
                latent=latent,
            )
            result.update(
                {
                    "latent_label": label,
                    "latent_description": description,
                    "latent_norm": float(torch.linalg.vector_norm(latent_cpu).item()),
                    "latent": latent_cpu.tolist(),
                }
            )
            results.append(result)
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in results)
    summary = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "demo": demo_name,
        "samples": len(results),
        "latent_scale": args.scale,
        "successes": successes,
        "success_rate": successes / max(1, len(results)),
        "avg_reward": float(np.mean([item["reward"] for item in results])) if results else 0.0,
        "avg_steps": float(np.mean([item["steps"] for item in results])) if results else 0.0,
        "results": results,
    }
    metrics_path = out_dir / "latent_sweep_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"} | {"metrics": str(metrics_path)}, indent=2))


def main() -> None:
    """Command-line entrypoint.

    The CLI keeps the whole demo path in one file: download data, train ACT,
    run a single rollout, or evaluate a batch of rollouts.
    """

    parser = argparse.ArgumentParser(description="Tiny ACT demo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="download a small public robomimic dataset")
    dl.add_argument("--dataset", choices=DATASETS.keys(), default="lift-ph")
    dl.add_argument("--out", type=Path)
    dl.add_argument("--force", action="store_true")

    ri = sub.add_parser("render-images", help="render raw robomimic states into image-observation HDF5")
    ri.add_argument("--data", required=True, help="raw robomimic demo_v15.hdf5 file")
    ri.add_argument("--out", required=True, help="output HDF5 with obs/<camera>_image and robot proprioception")
    ri.add_argument("--max-demos", type=int, default=20)
    ri.add_argument("--camera", default="agentview")
    ri.add_argument("--image-key", default="agentview_image")
    ri.add_argument("--width", type=int, default=128)
    ri.add_argument("--height", type=int, default=128)
    ri.add_argument("--max-steps", type=int, default=100)

    tr = sub.add_parser("train", help="train ACT on an HDF5 demo file")
    tr.add_argument("--data", required=True)
    tr.add_argument("--out", default="runs/smoke")
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--chunk-size", type=int, default=8)
    tr.add_argument("--dim", type=int, default=128)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--device", default="cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--max-demos", type=int, default=20, help="cap demos for a small local run")
    tr.add_argument("--max-minutes", type=float, default=None, help="stop training after roughly this many minutes")
    tr.add_argument("--resume", default=None, help="checkpoint to warm-start from; optimizer state starts fresh")
    tr.add_argument("--obs-mode", choices=["low_dim", "image"], default="low_dim", help="train from privileged low-dim state or image + robot proprioception")
    tr.add_argument("--image-key", default="agentview_image", help="robomimic image observation key for --obs-mode image")
    tr.add_argument("--vision-backbone", choices=["scratch_cnn", "resnet18"], default="scratch_cnn", help="image encoder used by --obs-mode image")
    tr.add_argument("--vision-pretrained", action="store_true", help="use pretrained torchvision weights when supported by the selected vision backbone")
    tr.add_argument("--freeze-vision", action="store_true", help="freeze the convolutional vision backbone and train only the ACT/projection layers")
    tr.add_argument("--eval-every-epochs", type=int, default=0, help="run closed-loop rollout eval every N training epochs; 0 disables")
    tr.add_argument("--eval-before-train", action="store_true", help="record one rollout eval before the first optimizer step")
    tr.add_argument("--eval-episodes", type=int, default=0, help="number of rollout episodes for each in-training eval")
    tr.add_argument("--eval-start-demo-index", type=int, default=0)
    tr.add_argument("--eval-camera", default="agentview")
    tr.add_argument("--eval-width", type=int, default=128)
    tr.add_argument("--eval-height", type=int, default=128)
    tr.add_argument("--eval-fps", type=int, default=20)
    tr.add_argument("--eval-max-steps", type=int, default=100)
    tr.add_argument("--eval-videos", type=int, default=0, help="number of rollout MP4s to save for each in-training eval")

    ro = sub.add_parser("rollout", help="run a trained ACT checkpoint in robosuite and save an mp4")
    ro.add_argument("--checkpoint", required=True)
    ro.add_argument("--data", required=True)
    ro.add_argument("--out", default="runs/rollout.mp4")
    ro.add_argument("--device", default="cpu")
    ro.add_argument("--demo-index", type=int, default=0, help="which demonstration initial state to start from")
    ro.add_argument("--camera", default="agentview")
    ro.add_argument("--width", type=int, default=256)
    ro.add_argument("--height", type=int, default=256)
    ro.add_argument("--fps", type=int, default=20)
    ro.add_argument("--max-steps", type=int, default=100)

    ev = sub.add_parser("evaluate", help="run many policy rollouts and write metrics plus sample mp4s")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--out-dir", default="runs/eval")
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--episodes", type=int, default=20)
    ev.add_argument("--videos", type=int, default=3, help="number of initial rollouts to save as mp4")
    ev.add_argument("--start-demo-index", type=int, default=0)
    ev.add_argument("--camera", default="agentview")
    ev.add_argument("--width", type=int, default=256)
    ev.add_argument("--height", type=int, default=256)
    ev.add_argument("--fps", type=int, default=20)
    ev.add_argument("--max-steps", type=int, default=100)

    ls = sub.add_parser("latent-sweep", help="sample fixed CVAE latents from one start state and save comparison rollouts")
    ls.add_argument("--checkpoint", required=True)
    ls.add_argument("--data", required=True)
    ls.add_argument("--out-dir", default="runs/latent_sweep")
    ls.add_argument("--device", default="cpu")
    ls.add_argument("--demo-index", type=int, default=0)
    ls.add_argument("--samples", type=int, default=8, help="number of random latent samples")
    ls.add_argument("--scale", type=float, default=1.0, help="standard deviation multiplier for sampled latents")
    ls.add_argument("--seed", type=int, default=0)
    ls.add_argument("--include-zero", action=argparse.BooleanOptionalAction, default=True)
    ls.add_argument("--videos", type=int, default=9, help="number of sweep rollouts to save as mp4")
    ls.add_argument("--camera", default="agentview")
    ls.add_argument("--width", type=int, default=256)
    ls.add_argument("--height", type=int, default=256)
    ls.add_argument("--fps", type=int, default=20)
    ls.add_argument("--max-steps", type=int, default=100)

    ph = sub.add_parser("plot-history", help="write an SVG loss curve and JSON summary from a training history")
    ph.add_argument("--run", required=True, help="run directory containing history.jsonl")
    ph.add_argument("--history", default=None, help="explicit history.jsonl path")
    ph.add_argument("--out", default=None, help="output SVG path")
    ph.add_argument("--summary", default=None, help="output summary JSON path")
    ph.add_argument("--title", default=None)

    args = parser.parse_args()
    if args.cmd == "download":
        download_dataset(args.dataset, args.out, args.force)
    elif args.cmd == "render-images":
        render_image_dataset(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "rollout":
        rollout(args)
    elif args.cmd == "plot-history":
        plot_history(args)
    elif args.cmd == "evaluate":
        evaluate(args)
    else:
        latent_sweep(args)


if __name__ == "__main__":
    main()
