"""Vision encoder for processing camera observations."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class VisionEncoder(nn.Module):
    """
    Vision encoder using ResNet backbone.

    Processes RGB images from multiple cameras and outputs visual features.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        num_cameras: int = 2,
        feature_dim: int = 512,
        freeze_backbone: bool = False,
    ):
        """
        Initialize vision encoder.

        Args:
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            pretrained: Whether to use ImageNet pretrained weights
            num_cameras: Number of camera views
            feature_dim: Output feature dimension per camera
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_cameras = num_cameras
        self.feature_dim = feature_dim

        # Create ResNet backbone (one per camera)
        self.camera_encoders = nn.ModuleList()
        for _ in range(num_cameras):
            encoder = self._create_resnet_encoder(
                backbone, pretrained, feature_dim, freeze_backbone
            )
            self.camera_encoders.append(encoder)

    def _create_resnet_encoder(
        self,
        backbone: str,
        pretrained: bool,
        feature_dim: int,
        freeze_backbone: bool,
    ) -> nn.Module:
        """Create a single ResNet encoder."""
        # Load pretrained ResNet
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            resnet_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            resnet_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            resnet_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove final FC layer
        modules = list(resnet.children())[:-1]  # Remove avgpool and fc
        encoder = nn.Sequential(*modules)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in encoder.parameters():
                param.requires_grad = False

        # Add projection head to get desired feature dimension
        if resnet_dim != feature_dim:
            projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(resnet_dim, feature_dim),
                nn.ReLU(),
            )
            encoder = nn.Sequential(encoder, projection)
        else:
            encoder = nn.Sequential(encoder, nn.Flatten())

        return encoder

    def forward(self, images: dict) -> torch.Tensor:
        """
        Forward pass through vision encoder.

        Args:
            images: Dict mapping camera names to image tensors
                   Each tensor has shape (batch, channels, height, width)

        Returns:
            Visual features of shape (batch, num_cameras * feature_dim)
        """
        # Get camera names in sorted order for consistency
        camera_names = sorted(images.keys())

        if len(camera_names) != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, got {len(camera_names)}"
            )

        # Encode each camera view
        camera_features = []
        for i, cam_name in enumerate(camera_names):
            img = images[cam_name]

            # Normalize images (ImageNet normalization)
            # Assuming input is in [0, 255] range
            if img.max() > 1.0:
                img = img / 255.0

            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
            img = (img - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

            # Encode
            features = self.camera_encoders[i](img)
            camera_features.append(features)

        # Concatenate features from all cameras
        visual_features = torch.cat(camera_features, dim=1)

        return visual_features


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax layer for extracting spatial features.

    Often used in visuomotor policies to get 2D keypoint features.
    """

    def __init__(self, height: int, width: int, channel: int):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel

        # Create coordinate meshgrid
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        pos_x = pos_x.reshape(height * width)
        pos_y = pos_y.reshape(height * width)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channel, height, width)

        Returns:
            Spatial features of shape (batch, channel * 2)
        """
        batch_size = x.shape[0]

        # Flatten spatial dimensions
        x = x.view(batch_size, self.channel, self.height * self.width)

        # Apply softmax over spatial dimension
        attention = torch.softmax(x, dim=2)

        # Compute expected coordinates
        expected_x = torch.sum(self.pos_x * attention, dim=2, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=2, keepdim=True)

        # Concatenate x and y coordinates
        expected_xy = torch.cat([expected_x, expected_y], dim=2)

        # Flatten to (batch, channel * 2)
        features = expected_xy.view(batch_size, self.channel * 2)

        return features


class VisionEncoderWithSpatialSoftmax(nn.Module):
    """
    Vision encoder with Spatial Softmax for extracting spatial keypoints.

    Alternative to global pooling, useful for tasks requiring spatial reasoning.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        num_cameras: int = 2,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.num_cameras = num_cameras

        # Create ResNet encoders without final pooling
        self.camera_encoders = nn.ModuleList()
        for _ in range(num_cameras):
            encoder = self._create_resnet_conv(backbone, pretrained, freeze_backbone)
            self.camera_encoders.append(encoder)

        # Get feature map dimensions
        # For ResNet18/34: 512 channels, for ResNet50: 2048 channels
        if backbone in ["resnet18", "resnet34"]:
            conv_channels = 512
        else:
            conv_channels = 2048

        # Spatial softmax layers (assuming 3x3 feature maps after ResNet)
        # This depends on input image size
        self.spatial_softmax = nn.ModuleList([
            SpatialSoftmax(height=3, width=3, channel=conv_channels)
            for _ in range(num_cameras)
        ])

        self.output_dim = num_cameras * conv_channels * 2

    def _create_resnet_conv(
        self, backbone: str, pretrained: bool, freeze_backbone: bool
    ) -> nn.Module:
        """Create ResNet encoder up to conv layers (no pooling)."""
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove avgpool and fc
        modules = list(resnet.children())[:-2]
        encoder = nn.Sequential(*modules)

        if freeze_backbone:
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder

    def forward(self, images: dict) -> torch.Tensor:
        """Forward pass with spatial softmax."""
        camera_names = sorted(images.keys())

        camera_features = []
        for i, cam_name in enumerate(camera_names):
            img = images[cam_name]

            # Normalize
            if img.max() > 1.0:
                img = img / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
            img = (img - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

            # Get conv features
            conv_features = self.camera_encoders[i](img)

            # Apply spatial softmax
            spatial_features = self.spatial_softmax[i](conv_features)
            camera_features.append(spatial_features)

        return torch.cat(camera_features, dim=1)
