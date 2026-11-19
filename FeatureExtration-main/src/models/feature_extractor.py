"""
src/models/feature_extractor.py
================================
UnderwaterFeatureExtractor — Main Model

Integrates:
  1. NoiseMapEstimator     → backscatter / noise map
  2. IlluminationMapEstimator → illumination map
  3. ColourDeviationEstimator → per-channel colour cast map
  4. UnderwaterBackbone    → ResNet/UNet + DegradationAwareAttention
  5. ClassifierHead        → optional downstream classification

Full forward pass:
    raw_image
        │
        ├─► NoiseEstimator  → noise_map
        ├─► IllumEstimator  → illum_map
        └─► ColourEstimator → colour_map
                │
                ▼
        UnderwaterBackbone(raw_image, noise_map, illum_map, colour_map)
                │
                ├─► enhanced_image   (3, H, W)
                ├─► features         (64, H/4, W/4)
                └─► stage_feats      [list of attended maps]
                        │
                        ▼
               ClassifierHead → logits (optional)
"""

from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .estimators import (
    NoiseMapEstimator,
    IlluminationMapEstimator,
    ColourDeviationEstimator,
)
from .backbone import UnderwaterBackbone


# ════════════════════════════════════════════════════════════════
#  Classifier Head
# ════════════════════════════════════════════════════════════════

class ClassifierHead(nn.Module):
    """
    Downstream classification head.
    Input:  (B, feature_dim, H', W')
    Output: (B, num_classes)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim:  int   = 512,
        dropout:     float = 0.5,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.pool(features)     # (B, C, 1, 1)
        return self.head(x)         # (B, num_classes)


# ════════════════════════════════════════════════════════════════
#  Main Model
# ════════════════════════════════════════════════════════════════

class UnderwaterFeatureExtractor(nn.Module):
    """
    End-to-end model for underwater low-light image enhancement
    and feature extraction.

    Args:
        backbone:        ResNet variant ("resnet18" | "resnet34" | "resnet50")
        pretrained:      Use ImageNet-pretrained backbone weights
        reduction_ratio: CBAM channel reduction
        spatial_kernel:  CBAM spatial attention kernel size
        num_classes:     If > 0, attaches a classifier head
        base_features:   Base channel count for estimators
    """

    def __init__(
        self,
        backbone:        str   = "resnet34",
        pretrained:      bool  = True,
        reduction_ratio: int   = 16,
        spatial_kernel:  int   = 7,
        num_classes:     int   = 0,        # 0 = no classifier
        base_features:   int   = 32,
        hidden_dim:      int   = 512,
        dropout:         float = 0.5,
    ):
        super().__init__()

        # ── Degradation estimators ────────────────────────────
        self.noise_estimator  = NoiseMapEstimator(
            in_channels=3, base_features=base_features
        )
        self.illum_estimator  = IlluminationMapEstimator(
            in_channels=3, base_features=base_features
        )
        self.colour_estimator = ColourDeviationEstimator(
            in_channels=3, hidden=base_features
        )

        # ── Backbone ──────────────────────────────────────────
        self.backbone = UnderwaterBackbone(
            backbone=backbone,
            pretrained=pretrained,
            reduction_ratio=reduction_ratio,
            spatial_kernel=spatial_kernel,
        )

        # ── Optional classifier ───────────────────────────────
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = ClassifierHead(
                feature_dim=self.backbone.feature_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.classifier = None

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) raw low-light underwater image
               Values in [0, 1] (normalised)

        Returns dict with keys:
            "enhanced"     : (B, 3, H, W)       — enhanced output image
            "features"     : (B, 64, H/4, W/4)  — deep feature map
            "noise_map"    : (B, 1, H, W)        — estimated noise map
            "illum_map"    : (B, 1, H, W)        — estimated illumination map
            "colour_map"   : (B, 3, H, W)        — colour deviation map
            "stage_feats"  : list of attention feature maps
            "logits"       : (B, num_classes)    — only if classifier attached
        """
        # ── Stage 2: Estimate degradation maps ───────────────
        noise_map  = self.noise_estimator(x)     # (B, 1, H, W)
        illum_map  = self.illum_estimator(x)     # (B, 1, H, W)
        colour_map = self.colour_estimator(x)    # (B, 3, H, W)

        # ── Stage 3: Feature extraction + enhancement ─────────
        enhanced, features, stage_feats = self.backbone(
            x, noise_map, illum_map, colour_map
        )

        out = {
            "enhanced":    enhanced,
            "features":    features,
            "noise_map":   noise_map,
            "illum_map":   illum_map,
            "colour_map":  colour_map,
            "stage_feats": stage_feats,
        }

        # ── Stage 4: Classification (optional) ────────────────
        if self.classifier is not None:
            out["logits"] = self.classifier(features)

        return out

    # ──────────────────────────────────────────────────────────
    #  Utility methods
    # ──────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self):
        """Freeze backbone encoder for fine-tuning decoders only."""
        for p in self.backbone.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.backbone.encoder.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def get_attention_maps(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method for visualisation.
        Returns all intermediate maps without gradients.
        """
        self.eval()
        out = self.forward(x)
        return {
            "noise_map":   out["noise_map"],
            "illum_map":   out["illum_map"],
            "colour_map":  out["colour_map"],
            "stage_feat1": out["stage_feats"][0].mean(dim=1, keepdim=True),
            "stage_feat2": out["stage_feats"][1].mean(dim=1, keepdim=True),
            "stage_feat3": out["stage_feats"][2].mean(dim=1, keepdim=True),
        }


# ════════════════════════════════════════════════════════════════
#  Model factory
# ════════════════════════════════════════════════════════════════

def build_model(config: dict) -> UnderwaterFeatureExtractor:
    """
    Build model from config dictionary.

    Args:
        config: parsed YAML config (see config/config.yaml)

    Returns:
        Instantiated UnderwaterFeatureExtractor
    """
    m = config["model"]
    return UnderwaterFeatureExtractor(
        backbone        = m["backbone"],
        pretrained      = m["pretrained"],
        reduction_ratio = m["attention"]["reduction_ratio"],
        spatial_kernel  = m["attention"]["kernel_size"],
        num_classes     = m["classifier"]["num_classes"],
        base_features   = m["base_features"],
        hidden_dim      = m["classifier"]["hidden_dim"],
        dropout         = m["classifier"]["dropout"],
    )
