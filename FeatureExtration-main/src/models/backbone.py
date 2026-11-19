"""
src/models/backbone.py
=======================
CNN Backbone — ResNet/UNet Hybrid with Attention

Architecture overview:
  - Encoder: pretrained ResNet (18/34/50) feature extractor
  - Skip connections: like UNet
  - Each encoder stage followed by DegradationAwareAttention
  - Decoder: progressive upsampling → enhanced image output

The backbone serves dual purpose:
  1. Feature extraction (deep embeddings for classification)
  2. Image reconstruction (enhanced output image)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .attention import CBAM, DegradationAwareAttention


# ════════════════════════════════════════════════════════════════
#  Helper blocks
# ════════════════════════════════════════════════════════════════

class ConvBnReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """
    Single decoder step:
      1. Upsample (bilinear × 2)
      2. Concatenate skip connection
      3. Two ConvBnReLU layers
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnReLU(in_ch, out_ch),
        )
        self.block = nn.Sequential(
            ConvBnReLU(out_ch + skip_ch, out_ch),
            ConvBnReLU(out_ch, out_ch),
        )

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Handle spatial size mismatch (edge case)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


# ════════════════════════════════════════════════════════════════
#  ResNet encoder wrapper
# ════════════════════════════════════════════════════════════════

class ResNetEncoder(nn.Module):
    """
    Wraps torchvision ResNet (18 / 34 / 50) and exposes
    intermediate feature maps at 4 scales:
        s1: stride 2   (H/2,  W/2)
        s2: stride 4   (H/4,  W/4)
        s3: stride 8   (H/8,  W/8)
        s4: stride 16  (H/16, W/16)

    Returns list [s1, s2, s3, s4] and optional s0 (stem output).
    """

    _CHANNELS = {
        "resnet18": [64, 64, 128, 256, 512],
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
    }

    def __init__(self, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.out_channels  = self._CHANNELS[backbone]

        weights = "DEFAULT" if pretrained else None
        net = getattr(tvm, backbone)(weights=weights)

        # Split ResNet into stages
        self.stem    = nn.Sequential(net.conv1, net.bn1, net.relu)  # /2
        self.maxpool = net.maxpool                                   # /2 more
        self.layer1  = net.layer1   # /4  total from input
        self.layer2  = net.layer2   # /8
        self.layer3  = net.layer3   # /16
        self.layer4  = net.layer4   # /32

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            s0:   stem features   (B, 64, H/2, W/2)
            skips: [s1, s2, s3, s4] at /4, /8, /16, /32
        """
        s0 = self.stem(x)           # (B, 64,   H/2,  W/2)
        x  = self.maxpool(s0)

        s1 = self.layer1(x)         # (B, C1,   H/4,  W/4)
        s2 = self.layer2(s1)        # (B, C2,   H/8,  W/8)
        s3 = self.layer3(s2)        # (B, C3,   H/16, W/16)
        s4 = self.layer4(s3)        # (B, C4,   H/32, W/32)

        return s0, [s1, s2, s3, s4]


# ════════════════════════════════════════════════════════════════
#  Full Backbone: Encoder + Attention + Decoder
# ════════════════════════════════════════════════════════════════

class UnderwaterBackbone(nn.Module):
    """
    ResNet/UNet Hybrid backbone.

    Forward pass produces:
      - enhanced:   (B, 3, H, W)   — reconstructed clean image
      - features:   (B, C, H', W') — deep feature map for downstream tasks
      - stage_feats: list of attended feature maps per encoder stage

    The attention at each stage is DegradationAwareAttention,
    conditioned on noise/illumination/colour maps from the estimators.
    """

    def __init__(
        self,
        backbone:        str  = "resnet34",
        pretrained:      bool = True,
        reduction_ratio: int  = 16,
        spatial_kernel:  int  = 7,
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.encoder   = ResNetEncoder(backbone, pretrained)
        enc_channels   = self.encoder.out_channels   # [64, C1, C2, C3, C4]

        # ── Degradation-Aware Attention at each encoder stage ─
        #    Applied after layers 1, 2, 3  (not 4 — too small)
        self.att1 = DegradationAwareAttention(enc_channels[1], reduction_ratio, spatial_kernel)
        self.att2 = DegradationAwareAttention(enc_channels[2], reduction_ratio, spatial_kernel)
        self.att3 = DegradationAwareAttention(enc_channels[3], reduction_ratio, spatial_kernel)
        # CBAM only (no prior needed) for deepest features
        self.att4 = CBAM(enc_channels[4], reduction_ratio, spatial_kernel)

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ConvBnReLU(enc_channels[4], 256),
            ConvBnReLU(256, 256),
        )

        # ── Decoder (UNet-style) ──────────────────────────────
        # dec4: 256 + s3 → 256
        self.dec4 = DecoderBlock(256,          enc_channels[3], 256)
        # dec3: 256 + s2 → 128
        self.dec3 = DecoderBlock(256,          enc_channels[2], 128)
        # dec2: 128 + s1 → 64
        self.dec2 = DecoderBlock(128,          enc_channels[1], 64)
        # dec1: 64  + s0 → 32
        self.dec1 = DecoderBlock(64,           enc_channels[0], 32)

        # ── Feature output ────────────────────────────────────
        # After dec2: 64-channel feature map at H/4
        self.feature_dim = 64

        # ── Image output head ─────────────────────────────────
        self.image_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnReLU(32, 16),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_decoder()

    def _init_decoder(self):
        """Initialize decoder weights (encoder is pretrained)."""
        for module in [
            self.bottleneck, self.dec4, self.dec3, self.dec2, self.dec1,
            self.image_head, self.att1, self.att2, self.att3, self.att4,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:          torch.Tensor,   # (B, 3, H, W)
        noise_map:  torch.Tensor,   # (B, 1, H, W) from NoiseMapEstimator
        illum_map:  torch.Tensor,   # (B, 1, H, W) from IlluminationMapEstimator
        colour_map: torch.Tensor,   # (B, 3, H, W) from ColourDeviationEstimator
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            enhanced:    (B, 3, H, W)  — enhanced output image
            features:    (B, 64, H/4, W/4) — for classifier / further processing
            stage_feats: [s1_att, s2_att, s3_att] — for visualisation
        """
        # ── Encode ──────────────────────────────────────────
        s0, (s1, s2, s3, s4) = self.encoder(x)

        # ── Attention at each stage ──────────────────────────
        s1_att = self.att1(s1, noise_map, illum_map, colour_map)
        s2_att = self.att2(s2, noise_map, illum_map, colour_map)
        s3_att = self.att3(s3, noise_map, illum_map, colour_map)
        s4_att = self.att4(s4)

        # ── Bottleneck ───────────────────────────────────────
        bn = self.bottleneck(s4_att)    # (B, 256, H/32, W/32)

        # ── Decode ───────────────────────────────────────────
        d4       = self.dec4(bn,   skip=s3_att)    # (B, 256, H/16, W/16)
        d3       = self.dec3(d4,   skip=s2_att)    # (B, 128, H/8,  W/8)
        features = self.dec2(d3,   skip=s1_att)    # (B, 64,  H/4,  W/4)
        d1       = self.dec1(features, skip=s0)    # (B, 32,  H/2,  W/2)

        # ── Output ───────────────────────────────────────────
        enhanced = self.image_head(d1)              # (B, 3,  H,    W)

        stage_feats = [s1_att, s2_att, s3_att]

        return enhanced, features, stage_feats
