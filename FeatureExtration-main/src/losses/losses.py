"""
src/losses/losses.py
=====================
Loss Functions for Underwater Image Enhancement

Implements:
  1. ReconstructionLoss        — pixel-level L1 + L2 (Charbonnier variant)
  2. PerceptualLoss            — VGG feature-space loss
  3. SSIMLoss                  — structural similarity
  4. ColourConsistencyLoss     — penalises per-channel mean deviation
  5. ContrastEnhancementLoss   — encourages higher local contrast
  6. NoiseSmoothLoss           — TV regularisation on noise map
  7. IllumSmoothLoss           — gradient smoothness on illumination map
  8. UnderwaterCombinedLoss    — weighted combination of all above

References:
  - Perceptual Losses: Johnson et al., ECCV 2016
  - SSIM: Wang et al., TIP 2004
  - Charbonnier: Charbonnier et al., ICIP 1994
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models import VGG16_Weights

from ..data.preprocessing import compute_laplacian_map


# ════════════════════════════════════════════════════════════════
#  1. Reconstruction Loss (Charbonnier)
# ════════════════════════════════════════════════════════════════

class ReconstructionLoss(nn.Module):
    """
    Charbonnier loss: sqrt((x-y)^2 + eps^2) — a smooth L1 variant
    that is differentiable at zero and robust to outliers.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))


# ════════════════════════════════════════════════════════════════
#  2. Perceptual Loss (VGG16)
# ════════════════════════════════════════════════════════════════

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 feature maps.
    Layers used: relu1_2, relu2_2, relu3_3 (low-mid level features).

    Frozen — not updated during training.
    """

    VGG_LAYERS = [4, 9, 16]   # relu1_2, relu2_2, relu3_3

    def __init__(self, device: Optional[str] = None):
        super().__init__()

        vgg = tvm.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slices = nn.ModuleList([
            nn.Sequential(*[vgg[i] for i in range(0, end)])
            for end in [4, 9, 16]  # up to relu1_2, relu2_2, relu3_3
        ])
        # Actually build three slices correctly
        layers = list(vgg.children())
        self.slice1 = nn.Sequential(*layers[:4])    # relu1_2
        self.slice2 = nn.Sequential(*layers[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*layers[9:16])  # relu3_3

        for param in self.parameters():
            param.requires_grad = False

        # VGG normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _extract(self, x: torch.Tensor):
        x = self._normalize(x)
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return h1, h2, h3

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred_feats   = self._extract(pred)
        target_feats = self._extract(target)
        loss = sum(
            F.l1_loss(p, t) for p, t in zip(pred_feats, target_feats)
        )
        return loss / len(pred_feats)


# ════════════════════════════════════════════════════════════════
#  3. SSIM Loss
# ════════════════════════════════════════════════════════════════

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss: L = 1 - SSIM(pred, target).
    Uses a Gaussian window.
    """

    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel     = channel
        self.register_buffer("window", self._create_window(window_size, channel))

    @staticmethod
    def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(window_size, dtype=torch.float32)
        x = x - window_size // 2
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1d = self._gaussian(window_size, 1.5)
        _2d = _1d.unsqueeze(1) * _1d.unsqueeze(0)
        window = _2d.unsqueeze(0).unsqueeze(0)
        return window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        w   = self.window.to(img1.device)
        mu1  = F.conv2d(img1, w, padding=pad, groups=self.channel)
        mu2  = F.conv2d(img2, w, padding=pad, groups=self.channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, w, padding=pad, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, w, padding=pad, groups=self.channel) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, w, padding=pad, groups=self.channel) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        return (num / den).mean()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return 1.0 - self._ssim(pred, target)


# ════════════════════════════════════════════════════════════════
#  4. Colour Consistency Loss
# ════════════════════════════════════════════════════════════════

class ColourConsistencyLoss(nn.Module):
    """
    Penalises per-channel mean deviation between prediction and target.
    Underwater colour cast (dominant blue/green) manifests as a large
    inter-channel mean gap — this loss closes it.

    Also penalises the colour deviation map from being far from zero
    (so the estimator learns meaningful values, not arbitrary ones).
    """

    def forward(
        self,
        pred:       torch.Tensor,   # (B, 3, H, W)
        target:     torch.Tensor,   # (B, 3, H, W)
        colour_map: Optional[torch.Tensor] = None,  # (B, 3, H, W)
    ) -> torch.Tensor:
        # Per-channel mean matching
        pred_mean   = pred.mean(dim=[2, 3])    # (B, 3)
        target_mean = target.mean(dim=[2, 3])  # (B, 3)
        loss = F.l1_loss(pred_mean, target_mean)

        # Colour map should be small (minimal residual deviation)
        if colour_map is not None:
            loss = loss + 0.1 * colour_map.abs().mean()

        return loss


# ════════════════════════════════════════════════════════════════
#  5. Contrast Enhancement Loss
# ════════════════════════════════════════════════════════════════

class ContrastEnhancementLoss(nn.Module):
    """
    Encourages the enhanced image to have at least as much
    local contrast as the target reference.

    Contrast is measured via Laplacian edge response.
    Loss = max(0, target_contrast - pred_contrast) + L1(edges)
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred_lap   = compute_laplacian_map(pred)     # (B, 1, H, W)
        target_lap = compute_laplacian_map(target)   # (B, 1, H, W)

        # Hinge: penalty if pred has less contrast than target
        hinge = F.relu(target_lap - pred_lap).mean()

        # Direct edge matching
        edge_match = F.l1_loss(pred_lap, target_lap)

        return hinge + 0.5 * edge_match


# ════════════════════════════════════════════════════════════════
#  6. Noise Map Smoothness Regularisation (TV)
# ════════════════════════════════════════════════════════════════

class NoiseSmoothLoss(nn.Module):
    """
    Total Variation regularisation on the noise map.
    Prevents the estimator from producing spatially erratic predictions.
    """

    def forward(self, noise_map: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(noise_map[:, :, 1:, :] - noise_map[:, :, :-1, :])
        dw = torch.abs(noise_map[:, :, :, 1:] - noise_map[:, :, :, :-1])
        return dh.mean() + dw.mean()


# ════════════════════════════════════════════════════════════════
#  7. Illumination Smoothness Loss
# ════════════════════════════════════════════════════════════════

class IllumSmoothLoss(nn.Module):
    """
    Gradient-weighted Total Variation on illumination map.
    Encourages the illumination map to be piecewise smooth —
    abrupt transitions should only occur at strong image gradients.
    """

    def forward(
        self, illum_map: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        # Image gradient magnitudes (used as edge-aware weights)
        with torch.no_grad():
            img_grey = image.mean(dim=1, keepdim=True)
            grad_h   = torch.abs(img_grey[:, :, 1:, :] - img_grey[:, :, :-1, :])
            grad_w   = torch.abs(img_grey[:, :, :, 1:] - img_grey[:, :, :, :-1])
            # Larger image gradient → we allow larger illumination gradient
            weight_h = torch.exp(-10.0 * grad_h)
            weight_w = torch.exp(-10.0 * grad_w)

        illum_h = torch.abs(illum_map[:, :, 1:, :] - illum_map[:, :, :-1, :])
        illum_w = torch.abs(illum_map[:, :, :, 1:] - illum_map[:, :, :, :-1])

        return (weight_h * illum_h).mean() + (weight_w * illum_w).mean()


# ════════════════════════════════════════════════════════════════
#  8. Combined Loss
# ════════════════════════════════════════════════════════════════

class UnderwaterCombinedLoss(nn.Module):
    """
    Weighted combination of all underwater-specific losses.

    Weights are taken from config["training"]["loss"].
    """

    def __init__(self, config: dict):
        super().__init__()
        w = config["training"]["loss"]

        self.w_recon      = w.get("reconstruction",      1.0)
        self.w_percep     = w.get("perceptual",           0.1)
        self.w_colour     = w.get("color_consistency",    0.5)
        self.w_contrast   = w.get("contrast",             0.3)
        self.w_ssim       = w.get("ssim",                 0.5)
        self.w_noise_reg  = w.get("noise_regularization", 0.01)
        self.w_illum_sm   = w.get("illumination_smooth",  0.01)

        self.recon    = ReconstructionLoss()
        self.percep   = PerceptualLoss()
        self.ssim_l   = SSIMLoss()
        self.colour_l = ColourConsistencyLoss()
        self.contrast = ContrastEnhancementLoss()
        self.noise_tv = NoiseSmoothLoss()
        self.illum_tv = IllumSmoothLoss()

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        target:    torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model_out: output dict from UnderwaterFeatureExtractor
            target:    (B, 3, H, W) reference clean image

        Returns:
            total_loss: scalar tensor
            loss_dict:  per-component losses (for logging)
        """
        pred       = model_out["enhanced"]
        noise_map  = model_out["noise_map"]
        illum_map  = model_out["illum_map"]
        colour_map = model_out["colour_map"]
        raw_input  = model_out.get("input", pred)  # fallback

        l_recon    = self.recon(pred, target)
        l_percep   = self.percep(pred, target)
        l_ssim     = self.ssim_l(pred, target)
        l_colour   = self.colour_l(pred, target, colour_map)
        l_contrast = self.contrast(pred, target)
        l_noise    = self.noise_tv(noise_map)
        l_illum    = self.illum_tv(illum_map, pred)

        total = (
            self.w_recon    * l_recon
          + self.w_percep   * l_percep
          + self.w_ssim     * l_ssim
          + self.w_colour   * l_colour
          + self.w_contrast * l_contrast
          + self.w_noise_reg * l_noise
          + self.w_illum_sm  * l_illum
        )

        log = {
            "loss/total":        total.item(),
            "loss/recon":        l_recon.item(),
            "loss/perceptual":   l_percep.item(),
            "loss/ssim":         l_ssim.item(),
            "loss/colour":       l_colour.item(),
            "loss/contrast":     l_contrast.item(),
            "loss/noise_tv":     l_noise.item(),
            "loss/illum_smooth": l_illum.item(),
        }

        return total, log


# ─────────────────────────────────────────────────────────────
#  Fix missing import in forward
# ─────────────────────────────────────────────────────────────
#from typing import Tuple   # already imported above; safe duplicate
