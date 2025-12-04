"""
UNet3D SDF Prediction Model using MONAI.

This is an independent model from AGQ that predicts 3D Signed Distance Function (SDF)
from EM images. The SDF represents the distance to the nearest boundary for each voxel:
- Negative values: inside instances
- Positive values: outside instances (background)
- Zero: on the boundary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import numpy as np

from monai.networks.nets import UNet
from monai.networks.layers import Norm


class SDFUNet3D(nn.Module):
    """
    3D UNet for SDF prediction using MONAI's UNet implementation.
    
    This model takes grayscale EM images as input and predicts the 3D SDF map.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale EM)
        out_channels: Number of output channels (default: 1 for SDF)
        channels: Feature channels at each level (default: [32, 64, 128, 256, 512])
        strides: Downsampling strides (default: [2, 2, 2, 2])
        num_res_units: Number of residual units per level
        norm: Normalization type ('batch', 'instance', 'group')
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        norm: str = 'instance',
        dropout: float = 0.0,
        spatial_dims: int = 3,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Map norm string to MONAI Norm enum
        norm_map = {
            'batch': Norm.BATCH,
            'instance': Norm.INSTANCE,
            'group': Norm.GROUP,
        }
        norm_type = norm_map.get(norm, Norm.INSTANCE)
        
        # Build MONAI UNet
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm_type,
            dropout=dropout,
        )
        
        # Final activation: tanh to constrain output to [-1, 1] range
        # (SDF is typically normalized to this range)
        self.output_activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor, apply_activation: bool = True) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            apply_activation: Whether to apply tanh activation
        
        Returns:
            SDF prediction of shape (B, 1, D, H, W)
        """
        out = self.unet(x)
        if apply_activation:
            out = self.output_activation(out)
        return out
    
    @staticmethod
    def parse_config(cfg, kwargs):
        """Parse config to get model kwargs (for compatibility with connectomics)."""
        # Can be extended to read from cfg if needed
        return kwargs


class SDFLoss(nn.Module):
    """
    Loss function for SDF prediction.
    
    Combines L1/L2 loss with optional boundary-aware weighting.
    """
    
    def __init__(
        self,
        loss_type: str = 'l1',
        boundary_weight: float = 2.0,
        boundary_threshold: float = 0.1,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.boundary_weight = boundary_weight
        self.boundary_threshold = boundary_threshold
        
        if loss_type == 'l1':
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted SDF loss.
        
        Args:
            pred: Predicted SDF (B, 1, D, H, W)
            target: Ground truth SDF (B, 1, D, H, W)
            weight: Optional weight mask (B, 1, D, H, W)
        
        Returns:
            Scalar loss value
        """
        loss = self.base_loss(pred, target)
        
        # Boundary-aware weighting: higher weight near boundaries (where SDF ≈ 0)
        if self.boundary_weight > 1.0:
            boundary_mask = (torch.abs(target) < self.boundary_threshold).float()
            boundary_weights = 1.0 + (self.boundary_weight - 1.0) * boundary_mask
            loss = loss * boundary_weights
        
        if weight is not None:
            loss = loss * weight
        
        return loss.mean()


class SDFGradientLoss(nn.Module):
    """
    Gradient consistency loss for SDF.
    
    Encourages the SDF gradient magnitude to be close to 1 (Eikonal constraint).
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute Eikonal loss: |∇SDF| should be close to 1.
        
        Args:
            pred: Predicted SDF (B, 1, D, H, W)
        
        Returns:
            Scalar loss value
        """
        # Compute gradients using finite differences
        grad_z = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        grad_y = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        grad_x = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        # Pad to same size
        grad_z = F.pad(grad_z, (0, 0, 0, 0, 0, 1))
        grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
        grad_x = F.pad(grad_x, (0, 1, 0, 0, 0, 0))
        
        # Gradient magnitude
        grad_mag = torch.sqrt(grad_z**2 + grad_y**2 + grad_x**2 + 1e-8)
        
        # Eikonal loss: (|∇SDF| - 1)^2
        eikonal_loss = ((grad_mag - 1.0) ** 2).mean()
        
        return self.weight * eikonal_loss


class CombinedSDFLoss(nn.Module):
    """
    Combined loss for SDF prediction.
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        gradient_weight: float = 0.1,
        boundary_weight: float = 2.0,
    ):
        super().__init__()
        self.sdf_loss = SDFLoss(
            loss_type='l1',
            boundary_weight=boundary_weight,
        )
        self.gradient_loss = SDFGradientLoss(weight=gradient_weight)
        self.l1_weight = l1_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual losses for logging
        """
        l1_loss = self.sdf_loss(pred, target, weight)
        grad_loss = self.gradient_loss(pred)
        
        total_loss = self.l1_weight * l1_loss + grad_loss
        
        loss_dict = {
            'sdf_l1': l1_loss.item(),
            'sdf_grad': grad_loss.item(),
            'sdf_total': total_loss.item(),
        }
        
        return total_loss, loss_dict


def build_sdf_model(cfg=None, **kwargs) -> SDFUNet3D:
    """
    Build SDF prediction model.
    
    Args:
        cfg: Optional config object
        **kwargs: Model arguments
    
    Returns:
        SDFUNet3D model
    """
    default_kwargs = {
        'in_channels': 1,
        'out_channels': 1,
        'channels': (32, 64, 128, 256, 512),
        'strides': (2, 2, 2, 2),
        'num_res_units': 2,
        'norm': 'instance',
        'dropout': 0.0,
    }
    
    # Override with provided kwargs
    default_kwargs.update(kwargs)
    
    # Parse from config if provided
    if cfg is not None and hasattr(cfg, 'MODEL'):
        if hasattr(cfg.MODEL, 'SDF_CHANNELS'):
            default_kwargs['channels'] = tuple(cfg.MODEL.SDF_CHANNELS)
        if hasattr(cfg.MODEL, 'SDF_STRIDES'):
            default_kwargs['strides'] = tuple(cfg.MODEL.SDF_STRIDES)
        if hasattr(cfg.MODEL, 'SDF_NORM'):
            default_kwargs['norm'] = cfg.MODEL.SDF_NORM
    
    return SDFUNet3D(**default_kwargs)


if __name__ == '__main__':
    # Test the model
    print("Testing SDFUNet3D...")
    
    model = SDFUNet3D(
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
    )
    
    # Test forward pass
    x = torch.randn(1, 1, 32, 64, 64)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    
    # Test loss
    loss_fn = CombinedSDFLoss()
    target = torch.randn(1, 1, 32, 64, 64).tanh()
    loss, loss_dict = loss_fn(y, target)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
