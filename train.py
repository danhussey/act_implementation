"""Training script for ACT model."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import yaml

from act_implementation.models.vision_encoder import VisionEncoder
from act_implementation.models.act_model import ACTModel
from act_implementation.data.dataset import create_dataloaders


def compute_loss(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 10.0,
) -> tuple:
    """
    Compute ACT training loss.

    Combines reconstruction loss (L2) with KL divergence for CVAE.

    Args:
        predicted_actions: Predicted action chunks (batch, chunk_size, action_dim)
        target_actions: Ground truth actions (batch, chunk_size, action_dim)
        mu: CVAE latent mean (batch, latent_dim)
        logvar: CVAE latent log variance (batch, latent_dim)
        kl_weight: Weight for KL divergence term

    Returns:
        total_loss, reconstruction_loss, kl_loss
    """
    # Reconstruction loss (L2)
    reconstruction_loss = nn.functional.mse_loss(predicted_actions, target_actions)

    # KL divergence loss
    # KL(q(z|x,c) || p(z)) where p(z) = N(0, I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    # Total loss
    total_loss = reconstruction_loss + kl_weight * kl_loss

    return total_loss, reconstruction_loss, kl_loss


def train_epoch(
    vision_encoder: nn.Module,
    act_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    kl_weight: float = 10.0,
    use_amp: bool = False,
) -> dict:
    """Train for one epoch."""
    vision_encoder.train()
    act_model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move to device
        states = batch["state"].to(device)
        images = {k: v.to(device) for k, v in batch["images"].items()}
        actions = batch["actions"].to(device)

        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp):
            visual_features = vision_encoder(images)
            predicted_actions, mu, logvar = act_model(visual_features, states, actions)

            # Compute loss
            loss, recon_loss, kl_loss = compute_loss(
                predicted_actions, actions, mu, logvar, kl_weight
            )

        # Backward pass
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

        pbar.set_postfix({
            "loss": loss.item(),
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
        })

    num_batches = len(train_loader)
    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon_loss / num_batches,
        "kl_loss": total_kl_loss / num_batches,
    }


@torch.no_grad()
def validate(
    vision_encoder: nn.Module,
    act_model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    kl_weight: float = 10.0,
    use_amp: bool = False,
) -> dict:
    """Validate the model."""
    vision_encoder.eval()
    act_model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    for batch in tqdm(val_loader, desc="Validation"):
        states = batch["state"].to(device)
        images = {k: v.to(device) for k, v in batch["images"].items()}
        actions = batch["actions"].to(device)

        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp):
            visual_features = vision_encoder(images)
            predicted_actions, mu, logvar = act_model(visual_features, states, actions)

            # Compute loss
            loss, recon_loss, kl_loss = compute_loss(
                predicted_actions, actions, mu, logvar, kl_weight
            )

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    num_batches = len(val_loader)
    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon_loss / num_batches,
        "kl_loss": total_kl_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ACT model")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=10, help="Action chunk size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl-weight", type=float, default=10.0, help="KL divergence weight")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Transformer hidden dim")
    parser.add_argument("--num-encoder-layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--num-decoder-layers", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--latent-dim", type=int, default=32, help="CVAE latent dimension")
    parser.add_argument("--device", type=str, default="mps", help="Device (cuda/mps/cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save-freq", type=int, default=10, help="Checkpoint save frequency")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision (CUDA only)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for speedup (PyTorch 2.0+)")
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # TensorBoard
    writer = SummaryWriter(log_dir="logs")

    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(
        data_path=args.data,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Get dataset info
    dataset = train_loader.dataset.dataset  # Unwrap from Subset
    stats = dataset.get_stats()
    print(f"Dataset: {stats['num_episodes']} episodes, {stats['num_samples']} samples")
    print(f"State dim: {stats['state_dim']}, Action dim: {stats['action_dim']}")

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
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        latent_dim=args.latent_dim,
    ).to(device)

    # Optimize models with torch.compile if requested (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling models with torch.compile...")
        vision_encoder = torch.compile(vision_encoder)
        act_model = torch.compile(act_model)

    # Enable mixed precision training
    use_amp = args.use_amp and device.type == "cuda"
    if use_amp:
        print("Using automatic mixed precision (AMP) training")

    # Optimizer
    params = list(vision_encoder.parameters()) + list(act_model.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Save config
    config = vars(args)
    config.update(stats)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(
            vision_encoder, act_model, train_loader, optimizer, device, args.kl_weight, use_amp
        )

        # Validate
        val_metrics = validate(
            vision_encoder, act_model, val_loader, device, args.kl_weight, use_amp
        )

        # Log metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"(recon: {train_metrics['recon_loss']:.4f}, kl: {train_metrics['kl_loss']:.4f})")
        print(f"Val Loss: {val_metrics['loss']:.4f} "
              f"(recon: {val_metrics['recon_loss']:.4f}, kl: {val_metrics['kl_loss']:.4f})")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch,
                "vision_encoder": vision_encoder.state_dict(),
                "act_model": act_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": config,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "vision_encoder": vision_encoder.state_dict(),
                "act_model": act_model.state_dict(),
                "config": config,
            }, best_path)
            print(f"New best model saved: {best_path}")

        # Step scheduler
        scheduler.step()

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
