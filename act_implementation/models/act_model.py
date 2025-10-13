"""Action Chunking Transformer (ACT) model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder for action distribution modeling.

    Encodes action chunks into a latent distribution conditioned on observations.
    """

    def __init__(
        self,
        action_dim: int,
        chunk_size: int,
        latent_dim: int = 32,
        condition_dim: int = 512,
        hidden_dim: int = 256,
    ):
        """
        Initialize CVAE.

        Args:
            action_dim: Dimensionality of single action
            chunk_size: Number of actions in a chunk
            latent_dim: Dimensionality of latent code
            condition_dim: Dimensionality of conditioning (from encoder)
            hidden_dim: Hidden layer dimensionality
        """
        super().__init__()

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        action_input_dim = action_dim * chunk_size

        # Encoder: q(z | x, c) where x is actions, c is condition
        self.encoder = nn.Sequential(
            nn.Linear(action_input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: p(x | z, c)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_input_dim),
        )

    def encode(self, actions: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode actions into latent distribution.

        Args:
            actions: Action chunk (batch, chunk_size, action_dim)
            condition: Conditioning from encoder (batch, condition_dim)

        Returns:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)
        """
        # Flatten actions
        actions_flat = actions.reshape(actions.size(0), -1)

        # Concatenate with condition
        x = torch.cat([actions_flat, condition], dim=1)

        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to action chunk.

        Args:
            z: Latent code (batch, latent_dim)
            condition: Conditioning (batch, condition_dim)

        Returns:
            actions: Predicted action chunk (batch, chunk_size, action_dim)
        """
        # Concatenate latent and condition
        x = torch.cat([z, condition], dim=1)

        # Decode
        actions_flat = self.decoder(x)

        # Reshape to action chunk
        actions = actions_flat.reshape(-1, self.chunk_size, self.action_dim)

        return actions

    def forward(
        self,
        actions: Optional[torch.Tensor],
        condition: torch.Tensor,
        sample_latent: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through CVAE.

        Args:
            actions: Ground truth actions (batch, chunk_size, action_dim) or None
            condition: Conditioning (batch, condition_dim)
            sample_latent: Whether to sample latent or use zeros (for inference)

        Returns:
            decoded_actions: Predicted actions (batch, chunk_size, action_dim)
            mu: Latent mean (or None if actions is None)
            logvar: Latent log variance (or None if actions is None)
        """
        if actions is not None:
            # Training: encode actions
            mu, logvar = self.encode(actions, condition)
            z = self.reparameterize(mu, logvar)
        else:
            # Inference: sample from prior or use zeros
            mu, logvar = None, None
            batch_size = condition.size(0)
            if sample_latent:
                z = torch.randn(batch_size, self.latent_dim, device=condition.device)
            else:
                z = torch.zeros(batch_size, self.latent_dim, device=condition.device)

        # Decode
        decoded_actions = self.decode(z, condition)

        return decoded_actions, mu, logvar


class ACTModel(nn.Module):
    """
    Action Chunking Transformer (ACT) model.

    Combines vision encoder, transformer encoder-decoder, and CVAE for
    predicting action chunks from visual observations and robot state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        visual_feature_dim: int,
        chunk_size: int = 10,
        hidden_dim: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        latent_dim: int = 32,
    ):
        """
        Initialize ACT model.

        Args:
            state_dim: Dimensionality of robot state
            action_dim: Dimensionality of action
            visual_feature_dim: Dimensionality of visual features from vision encoder
            chunk_size: Number of actions to predict in chunk
            hidden_dim: Hidden dimension of transformer
            num_encoder_layers: Number of transformer encoder layers
            num_decoder_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            latent_dim: Dimensionality of CVAE latent code
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # Embed visual features and state into transformer dimension
        self.visual_embedding = nn.Linear(visual_feature_dim, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=chunk_size, dropout=dropout)

        # Transformer encoder (processes current observation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder (predicts action sequence)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Learnable query tokens for action chunk prediction
        self.action_queries = nn.Parameter(torch.randn(chunk_size, 1, hidden_dim))

        # CVAE for action distribution
        self.cvae = CVAE(
            action_dim=action_dim,
            chunk_size=chunk_size,
            latent_dim=latent_dim,
            condition_dim=hidden_dim,
            hidden_dim=256,
        )

        # Project decoder output to condition dimension
        self.decoder_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        sample_latent: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through ACT model.

        Args:
            visual_features: Visual features from encoder (batch, visual_feature_dim)
            state: Robot state (batch, state_dim)
            actions: Ground truth action chunks for training (batch, chunk_size, action_dim)
            sample_latent: Whether to sample CVAE latent (for inference)

        Returns:
            predicted_actions: Predicted action chunk (batch, chunk_size, action_dim)
            mu: CVAE latent mean (or None)
            logvar: CVAE latent log variance (or None)
        """
        batch_size = state.size(0)

        # Embed visual features and state
        visual_embed = self.visual_embedding(visual_features)  # (batch, hidden_dim)
        state_embed = self.state_embedding(state)  # (batch, hidden_dim)

        # Concatenate as encoder input sequence: [visual, state]
        encoder_input = torch.stack([visual_embed, state_embed], dim=0)  # (2, batch, hidden_dim)

        # Apply positional encoding (optional for encoder input)
        # encoder_input = self.pos_encoder(encoder_input)

        # Transformer encoder
        memory = self.transformer_encoder(encoder_input)  # (2, batch, hidden_dim)

        # Prepare decoder queries (learnable action queries)
        action_queries = self.action_queries.expand(-1, batch_size, -1)  # (chunk_size, batch, hidden_dim)
        action_queries = self.pos_encoder(action_queries)

        # Transformer decoder
        decoder_output = self.transformer_decoder(action_queries, memory)  # (chunk_size, batch, hidden_dim)

        # Pool decoder output to get conditioning for CVAE
        # Use the first query output as the condition
        condition = self.decoder_projection(decoder_output[0])  # (batch, hidden_dim)

        # CVAE to predict action chunk
        predicted_actions, mu, logvar = self.cvae(actions, condition, sample_latent)

        return predicted_actions, mu, logvar

    def get_action(
        self,
        visual_features: torch.Tensor,
        state: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Get action chunk for inference (with optional ensembling).

        Args:
            visual_features: Visual features (batch, visual_feature_dim)
            state: Robot state (batch, state_dim)
            num_samples: Number of samples to average over

        Returns:
            Action chunk (batch, chunk_size, action_dim)
        """
        self.eval()
        with torch.no_grad():
            if num_samples == 1:
                actions, _, _ = self.forward(visual_features, state, actions=None, sample_latent=True)
                return actions
            else:
                # Sample multiple times and average
                action_samples = []
                for _ in range(num_samples):
                    actions, _, _ = self.forward(visual_features, state, actions=None, sample_latent=True)
                    action_samples.append(actions)

                return torch.stack(action_samples).mean(dim=0)
