"""
src/models/attention.py
========================
Attention Modules

Implements:
  - Channel Attention (SE / CBAM-style)
  - Spatial Attention (CBAM-style)
  - CBAM (Convolutional Block Attention Module) — Woo et al. ECCV 2018
  - DegradationAwareAttention — CBAM guided by noise + illumination + colour maps
  - ECA (Efficient Channel Attention) — Wang et al. CVPR 2020

All modules accept (B, C, H, W) and return attended (B, C, H, W).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
#  Channel Attention
# ════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """
    CBAM channel attention: squeeze via both avg-pool and max-pool,
    then excite via a shared MLP.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction_ratio, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale   = self.sigmoid(avg_out + max_out)   # (B, C, 1, 1)
        return x * scale


# ════════════════════════════════════════════════════════════════
#  Spatial Attention
# ════════════════════════════════════════════════════════════════

class SpatialAttention(nn.Module):
    """
    CBAM spatial attention: compress channels via avg + max,
    then produce a spatial weight map.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv   = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        max_out, _ = x.max(dim=1, keepdim=True)         # (B, 1, H, W)
        descriptor  = torch.cat([avg_out, max_out], dim=1)   # (B, 2, H, W)
        scale = self.sigmoid(self.conv(descriptor))      # (B, 1, H, W)
        return x * scale


# ════════════════════════════════════════════════════════════════
#  CBAM — Full module (channel then spatial)
# ════════════════════════════════════════════════════════════════

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies channel attention then spatial attention sequentially.

    Reference: Woo et al. "CBAM: Convolutional Block Attention Module", ECCV 2018.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel: int  = 7,
    ):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ════════════════════════════════════════════════════════════════
#  Efficient Channel Attention (ECA-Net)
# ════════════════════════════════════════════════════════════════

class ECA(nn.Module):
    """
    Efficient Channel Attention — avoids dimensionality reduction,
    uses a 1-D convolution over the channel axis instead.

    Reference: Wang et al. "ECA-Net", CVPR 2020.
    """

    def __init__(self, in_channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size
        t = int(abs((math.log2(in_channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d   = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                          # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)           # (B, 1, C)
        y = self.conv1d(y)                            # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)         # (B, C, 1, 1)
        scale = self.sigmoid(y)
        return x * scale


# ════════════════════════════════════════════════════════════════
#  Degradation-Aware Attention  (underwater-specific)
# ════════════════════════════════════════════════════════════════

class DegradationAwareAttention(nn.Module):
    """
    Attention module conditioned on three degradation priors:
      - noise_map:  (B, 1, H, W)  — backscatter / noise level map
      - illum_map:  (B, 1, H, W)  — illumination level map
      - colour_map: (B, 3, H, W)  — colour deviation from neutral grey

    These maps are predicted by the noise and illumination estimators.
    The module fuses them into a spatial attention weight that biases
    the network toward the most degraded (and therefore most important
    for enhancement) regions.

    Architecture:
        [noise | illum | colour | feat_pool] → 5 channels
        → 3×3 conv → BN → ReLU → 1×1 conv → Sigmoid
        → Multiply with feature map
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel:  int = 7,
    ):
        super().__init__()

        # Prior fusion: noise(1) + illum(1) + colour(3) = 5 channels
        prior_channels = 5

        # Fuse degradation priors into a spatial attention weight
        self.prior_conv = nn.Sequential(
            nn.Conv2d(prior_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Standard CBAM on the feature map
        self.cbam = CBAM(in_channels, reduction_ratio, spatial_kernel)

        # Blend gate (learnable scalar)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x:          torch.Tensor,   # (B, C, H, W) feature map
        noise_map:  torch.Tensor,   # (B, 1, H, W)
        illum_map:  torch.Tensor,   # (B, 1, H, W)
        colour_map: torch.Tensor,   # (B, 3, H, W)
    ) -> torch.Tensor:

        # Resize priors to match feature map resolution
        H, W = x.shape[-2:]
        noise_r  = F.interpolate(noise_map,  size=(H, W), mode="bilinear", align_corners=False)
        illum_r  = F.interpolate(illum_map,  size=(H, W), mode="bilinear", align_corners=False)
        colour_r = F.interpolate(colour_map, size=(H, W), mode="bilinear", align_corners=False)

        # Compute prior-guided spatial weight
        priors     = torch.cat([noise_r, illum_r, colour_r], dim=1)  # (B, 5, H, W)
        prior_attn = self.prior_conv(priors)                          # (B, 1, H, W)

        # CBAM on features
        x_cbam = self.cbam(x)

        # Blend: mix prior-guided and CBAM attention
        alpha = torch.sigmoid(self.alpha)
        return alpha * (x_cbam * prior_attn) + (1.0 - alpha) * x_cbam


# ════════════════════════════════════════════════════════════════
#  Factory
# ════════════════════════════════════════════════════════════════

def build_attention(
    attention_type: str,
    in_channels:    int,
    reduction_ratio: int = 16,
    spatial_kernel:  int = 7,
) -> nn.Module:
    """
    Factory for attention modules.

    Args:
        attention_type: "cbam" | "eca" | "se"
        in_channels:    number of feature channels
    """
    attention_type = attention_type.lower()
    if attention_type == "cbam":
        return CBAM(in_channels, reduction_ratio, spatial_kernel)
    elif attention_type == "eca":
        return ECA(in_channels)
    elif attention_type == "degradation":
        return DegradationAwareAttention(in_channels, reduction_ratio, spatial_kernel)
    else:
        raise ValueError(
            f"Unknown attention type '{attention_type}'. "
            f"Choose from: cbam | eca | degradation"
        )
