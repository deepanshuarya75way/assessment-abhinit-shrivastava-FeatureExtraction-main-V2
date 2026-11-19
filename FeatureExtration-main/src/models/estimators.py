"""
src/models/estimators.py
=========================
Underwater-Aware Map Estimators

1. NoiseMapEstimator
   - Predicts a per-pixel noise level map (B, 1, H, W)
   - Designed to capture backscatter, sensor noise, and haze
   - Architecture: encoder–decoder with skip connections + residual blocks

2. IlluminationMapEstimator
   - Predicts a per-pixel illumination level map (B, 1, H, W)
   - Accounts for non-uniform lighting common in underwater scenes
   - Architecture: multi-scale dilated encoder → smooth decoder

3. ColourDeviationMap
   - Predicts a colour deviation map (B, 3, H, W) showing per-channel anomalies
   - Identifies the blue/green cast specific to underwater imaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
#  Shared building blocks
# ════════════════════════════════════════════════════════════════

class ConvBnReLU(nn.Sequential):
    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        kernel:  int  = 3,
        stride:  int  = 1,
        padding: int  = 1,
        dilation: int = 1,
        groups:  int  = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_ch, out_ch, kernel,
                stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """Basic residual block with two ConvBnReLU layers."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            ConvBnReLU(channels, channels, dilation=dilation,
                       padding=dilation),
            nn.Conv2d(channels, channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.body(x))


class UpConv(nn.Module):
    """Bilinear upsample + conv (avoids checkerboard artefacts)."""

    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.conv  = ConvBnReLU(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return self.conv(x)


# ════════════════════════════════════════════════════════════════
#  1. Noise Map Estimator
# ════════════════════════════════════════════════════════════════

class NoiseMapEstimator(nn.Module):
    """
    Predicts a single-channel noise map (B, 1, H, W) in [0, 1].

    High values → high noise / backscatter (regions needing denoising).
    Low values  → relatively clean regions.

    Architecture:
        Encoder: 3 down-sampling stages capturing multi-scale noise patterns
        Bottleneck: 2 residual blocks
        Decoder:  3 up-sampling stages (skip connections) → sigmoid output

    Parameters (~580K)
    """

    def __init__(
        self,
        in_channels:     int = 3,
        base_features:   int = 32,
    ):
        super().__init__()

        f = base_features  # shorthand

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = nn.Sequential(
            ConvBnReLU(in_channels, f),
            ConvBnReLU(f, f),
        )   # (B, f, H, W)

        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBnReLU(f, f * 2),
            ConvBnReLU(f * 2, f * 2),
        )   # (B, 2f, H/2, W/2)

        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBnReLU(f * 2, f * 4),
            ConvBnReLU(f * 4, f * 4),
        )   # (B, 4f, H/4, W/4)

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBnReLU(f * 4, f * 8),
            ResBlock(f * 8),
            ResBlock(f * 8),
        )   # (B, 8f, H/8, W/8)

        # ── Decoder ──────────────────────────────────────────
        self.up3 = UpConv(f * 8, f * 4)
        self.dec3 = ConvBnReLU(f * 8, f * 4)   # after skip concat

        self.up2 = UpConv(f * 4, f * 2)
        self.dec2 = ConvBnReLU(f * 4, f * 2)

        self.up1 = UpConv(f * 2, f)
        self.dec1 = ConvBnReLU(f * 2, f)

        # ── Output head ───────────────────────────────────────
        self.out_conv = nn.Sequential(
            nn.Conv2d(f, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) raw degraded image

        Returns:
            noise_map: (B, 1, H, W) noise level in [0, 1]
        """
        e1 = self.enc1(x)           # (B, f,   H,   W)
        e2 = self.enc2(e1)          # (B, 2f,  H/2, W/2)
        e3 = self.enc3(e2)          # (B, 4f,  H/4, W/4)
        bn = self.bottleneck(e3)    # (B, 8f,  H/8, W/8)

        d3 = self.up3(bn)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)    # (B, 1, H, W)


# ════════════════════════════════════════════════════════════════
#  2. Illumination Map Estimator
# ════════════════════════════════════════════════════════════════

class IlluminationMapEstimator(nn.Module):
    """
    Predicts a single-channel illumination map (B, 1, H, W) in [0, 1].

    High values → bright / well-lit regions.
    Low values  → dark / underexposed regions (needing enhancement).

    Architecture:
        Multi-scale dilated encoder (captures global light distribution)
        → Smooth Gaussian-like decoder
        → Sigmoid output

    Uses dilated convolutions to capture large-scale illumination patterns
    without losing spatial resolution at shallower features.

    Parameters (~320K)
    """

    def __init__(
        self,
        in_channels:   int = 3,
        base_features: int = 32,
    ):
        super().__init__()
        f = base_features

        # ── Dilated multi-scale encoder ───────────────────────
        self.stem = ConvBnReLU(in_channels, f)

        self.dilated_block = nn.ModuleList([
            ConvBnReLU(f, f, dilation=1,  padding=1),
            ConvBnReLU(f, f, dilation=2,  padding=2),
            ConvBnReLU(f, f, dilation=4,  padding=4),
            ConvBnReLU(f, f, dilation=8,  padding=8),
        ])
        # Fuse 4 dilated streams → f channels
        self.fuse = ConvBnReLU(f * 4, f * 2)

        # ── Down-sampling pathway ─────────────────────────────
        self.down1 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBnReLU(f * 2, f * 4),
            ResBlock(f * 4),
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBnReLU(f * 4, f * 8),
            ResBlock(f * 8),
        )

        # ── Up-sampling pathway (smooth) ──────────────────────
        self.up2   = UpConv(f * 8, f * 4)
        self.mix2  = ConvBnReLU(f * 8, f * 4)

        self.up1   = UpConv(f * 4, f * 2)
        self.mix1  = ConvBnReLU(f * 4, f * 2)

        # ── Output: smooth illumination map ───────────────────
        self.out_conv = nn.Sequential(
            nn.Conv2d(f * 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) raw degraded image

        Returns:
            illum_map: (B, 1, H, W) illumination level in [0, 1]
        """
        stem = self.stem(x)                         # (B, f, H, W)

        # Multi-scale dilated features
        dilated = torch.cat(
            [d(stem) for d in self.dilated_block], dim=1
        )                                           # (B, 4f, H, W)
        fused = self.fuse(dilated)                  # (B, 2f, H, W)

        d1 = self.down1(fused)                      # (B, 4f, H/2, W/2)
        d2 = self.down2(d1)                         # (B, 8f, H/4, W/4)

        u2 = self.up2(d2)
        u2 = self.mix2(torch.cat([u2, d1], dim=1))

        u1 = self.up1(u2)
        u1 = self.mix1(torch.cat([u1, fused], dim=1))

        return self.out_conv(u1)                    # (B, 1, H, W)


# ════════════════════════════════════════════════════════════════
#  3. Colour Deviation Map
# ════════════════════════════════════════════════════════════════

class ColourDeviationEstimator(nn.Module):
    """
    Estimates a 3-channel colour deviation map (B, 3, H, W).

    Values close to 0 → neutral colour (no cast).
    Positive blue/green channels with negative red → typical underwater cast.

    Lightweight: 3 conv layers only.
    """

    def __init__(self, in_channels: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnReLU(in_channels, hidden),
            ConvBnReLU(hidden, hidden),
            nn.Conv2d(hidden, 3, kernel_size=1),
            nn.Tanh(),  # output in [-1, +1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 3, H, W) in [-1, 1]
