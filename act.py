"""Small ACT training script for HDF5 robot demonstrations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


def write_mock_data(path: Path, episodes: int, steps: int, image_size: int) -> None:
    """Write a tiny synthetic dataset with the same shape as the real demos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for ep in range(episodes):
            group = f.create_group(f"episode_{ep}")
            states = rng.normal(size=(steps, 14)).astype(np.float32)
            actions = rng.normal(scale=0.3, size=(steps, 7)).astype(np.float32)
            group.create_dataset("states", data=states)
            group.create_dataset("actions", data=actions)
            group.create_dataset("images_agentview", data=rng.integers(0, 255, (steps, 3, image_size, image_size), dtype=np.uint8))
            group.create_dataset("images_wrist", data=rng.integers(0, 255, (steps, 3, image_size, image_size), dtype=np.uint8))
        f.attrs["num_episodes"] = episodes


class DemoDataset(Dataset):
    """Lazy HDF5 loader for action chunks."""

    def __init__(self, path: Path, chunk_size: int, episodes: list[int] | None = None, stats: dict | None = None):
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.file: h5py.File | None = None
        self.indices: list[tuple[int, int]] = []

        with h5py.File(self.path, "r") as f:
            total = int(f.attrs["num_episodes"])
            self.episodes = list(range(total)) if episodes is None else episodes
            first = f[f"episode_{self.episodes[0]}"]
            self.cameras = sorted(k.removeprefix("images_") for k in first if k.startswith("images_"))
            self.state_dim = first["states"].shape[1]
            self.action_dim = first["actions"].shape[1]
            for ep in self.episodes:
                length = f[f"episode_{ep}"]["actions"].shape[0]
                self.indices.extend((ep, t) for t in range(max(0, length - chunk_size + 1)))

        if stats is None:
            self.stats = self._compute_stats()
        else:
            self.stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in stats.items()}

    def _compute_stats(self) -> dict[str, torch.Tensor]:
        states, actions = [], []
        with h5py.File(self.path, "r") as f:
            for ep in self.episodes:
                group = f[f"episode_{ep}"]
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
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        return self.file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        ep, t = self.indices[idx]
        group = self._h5()[f"episode_{ep}"]
        state = torch.tensor(group["states"][t], dtype=torch.float32)
        actions = torch.tensor(group["actions"][t : t + self.chunk_size], dtype=torch.float32)
        images = [torch.tensor(group[f"images_{cam}"][t], dtype=torch.float32) / 255.0 for cam in self.cameras]
        return {
            "state": (state - self.stats["state_mean"]) / self.stats["state_std"],
            "actions": (actions - self.stats["action_mean"]) / self.stats["action_std"],
            "images": torch.cat(images, dim=0),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "state": torch.stack([x["state"] for x in batch]),
        "actions": torch.stack([x["actions"] for x in batch]),
        "images": torch.stack([x["images"] for x in batch]),
    }


def split_loaders(path: Path, chunk_size: int, batch_size: int, seed: int) -> tuple[DemoDataset, DataLoader, DataLoader]:
    full = DemoDataset(path, chunk_size)
    rng = np.random.default_rng(seed)
    episodes = np.array(full.episodes)
    rng.shuffle(episodes)
    cut = max(1, int(0.9 * len(episodes)))
    train_eps = episodes[:cut].tolist()
    val_eps = episodes[cut:].tolist() or train_eps
    train_set = DemoDataset(path, chunk_size, train_eps, stats={k: v.tolist() for k, v in full.stats.items()})
    val_set = DemoDataset(path, chunk_size, val_eps, stats={k: v.tolist() for k, v in full.stats.items()})
    return full, DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate), DataLoader(val_set, batch_size, collate_fn=collate)


class PositionalEncoding(nn.Module):
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


class ACT(nn.Module):
    def __init__(self, image_channels: int, state_dim: int, action_dim: int, chunk_size: int, dim: int = 128, latent_dim: int = 16):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.obs = nn.Linear(64 + state_dim, dim)
        self.latent = nn.Sequential(nn.Linear(chunk_size * action_dim + dim, dim), nn.ReLU(), nn.Linear(dim, latent_dim * 2))
        self.z_proj = nn.Linear(latent_dim, dim)
        self.query = nn.Parameter(torch.randn(chunk_size, 1, dim))
        self.pos = PositionalEncoding(dim, chunk_size)
        enc = nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True)
        dec = nn.TransformerDecoderLayer(dim, 4, dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, 2)
        self.decoder = nn.TransformerDecoder(dec, 2)
        self.head = nn.Linear(dim, action_dim)

    def forward(self, images: torch.Tensor, state: torch.Tensor, actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = state.size(0)
        obs = self.obs(torch.cat([self.cnn(images), state], dim=1))
        memory = self.encoder(obs.unsqueeze(1))
        if actions is None:
            mu = logvar = torch.zeros(batch, self.latent_dim, device=state.device)
            z = mu
        else:
            latent = self.latent(torch.cat([actions.flatten(1), obs], dim=1))
            mu, logvar = latent.chunk(2, dim=1)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        query = self.pos(self.query.expand(-1, batch, -1)).transpose(0, 1)
        query = query + self.z_proj(z).unsqueeze(1)
        return self.head(self.decoder(query, memory)), mu, logvar


def loss_fn(pred: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:
    recon = nn.functional.mse_loss(pred, target)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon + beta * kl


def run_epoch(model: ACT, loader: DataLoader, device: torch.device, optimizer=None, beta: float = 10.0) -> float:
    model.train(optimizer is not None)
    total = 0.0
    for batch in tqdm(loader, leave=False):
        images = batch["images"].to(device)
        states = batch["state"].to(device)
        actions = batch["actions"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            pred, mu, logvar = model(images, states, actions)
            loss = loss_fn(pred, actions, mu, logvar, beta)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full, train_loader, val_loader = split_loaders(Path(args.data), args.chunk_size, args.batch_size, args.seed)
    image_channels = len(full.cameras) * 3
    model = ACT(image_channels, full.state_dim, full.action_dim, args.chunk_size, args.dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        print(f"epoch={epoch + 1} train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "stats": {k: v.tolist() for k, v in full.stats.items()}, "args": vars(args)}, out / "best.pt")
    (out / "metrics.json").write_text(json.dumps({"best_val_loss": best, "cameras": full.cameras}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny ACT demo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    mock = sub.add_parser("mock", help="write synthetic HDF5 demos")
    mock.add_argument("--out", type=Path, default=Path("data/mock.hdf5"))
    mock.add_argument("--episodes", type=int, default=8)
    mock.add_argument("--steps", type=int, default=32)
    mock.add_argument("--image-size", type=int, default=64)

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

    args = parser.parse_args()
    if args.cmd == "mock":
        write_mock_data(args.out, args.episodes, args.steps, args.image_size)
    else:
        train(args)


if __name__ == "__main__":
    main()
