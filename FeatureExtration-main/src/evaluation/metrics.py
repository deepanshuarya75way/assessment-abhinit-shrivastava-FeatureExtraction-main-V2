"""
src/evaluation/metrics.py
==========================
Evaluation Metrics for Underwater Images

1. PSNR  — Peak Signal-to-Noise Ratio
           Standard image quality, measures fidelity to reference.

2. SSIM  — Structural Similarity Index
           Perceptually weighted: luminance, contrast, structure.

3. UIQM  — Underwater Image Quality Measure (Panetta et al., 2016)
           Specifically designed for underwater; no reference needed.
           UIQM = c1*UICM + c2*UISM + c3*UIConM

4. UCIQE — Underwater Colour Image Quality Evaluation (Yang et al., 2015)
           Linear combination of chroma std, luminance contrast, saturation.

WHY THESE METRICS?
  PSNR/SSIM require a reference. In real underwater scenarios there is
  often NO clean reference — hence UIQM and UCIQE (no-reference metrics)
  are critical for evaluating on real-world data.
"""

import math
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
#  PSNR
# ════════════════════════════════════════════════════════════════

def compute_psnr(
    pred:   torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher is better. Typical range for enhanced underwater: 20–35 dB.

    Args:
        pred, target: (B, 3, H, W) float32 in [0, max_val]
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)


def compute_psnr_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR on uint8 numpy arrays (0–255)."""
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(255.0 ** 2 / mse)


# ════════════════════════════════════════════════════════════════
#  SSIM
# ════════════════════════════════════════════════════════════════

def compute_ssim(
    pred:   torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> float:
    """
    Mean SSIM over batch.
    Returns scalar in [-1, 1]; 1 = identical.
    """
    from ..losses.losses import SSIMLoss
    ssim_module = SSIMLoss(window_size=window_size)
    ssim_loss   = ssim_module(pred.cpu(), target.cpu())
    return (1.0 - ssim_loss.item())   # loss = 1 - SSIM


def compute_ssim_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """SSIM via scikit-image for uint8 images."""
    from skimage.metrics import structural_similarity as sk_ssim
    return sk_ssim(pred, target, channel_axis=-1, data_range=255)


# ════════════════════════════════════════════════════════════════
#  UIQM — Underwater Image Quality Measure
# ════════════════════════════════════════════════════════════════

def _uicm(image: np.ndarray) -> float:
    """
    Underwater Image Colorfulness Measure.
    Measures asymmetry in red-blue and yellow-blue colour difference channels.
    """
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    RG = R.astype(np.float32) - G.astype(np.float32)
    YB = (R.astype(np.float32) + G.astype(np.float32)) / 2.0 - B.astype(np.float32)

    def _mean_diff(channel):
        mu    = channel.mean()
        sigma = channel.std()
        return -0.0268 * np.sqrt(mu ** 2 + sigma ** 2) + 0.1586

    return _mean_diff(RG) + _mean_diff(YB)


def _uism(image: np.ndarray) -> float:
    """
    Underwater Image Sharpness Measure.
    Evaluates edge strength in each colour channel using Sobel operator.
    """
    total = 0.0
    weights = [0.299, 0.587, 0.114]   # luminance weights per channel
    for c, w in zip(range(3), weights):
        ch     = image[:, :, c].astype(np.float32)
        sob_x  = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        sob_y  = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        emea   = np.sqrt(sob_x ** 2 + sob_y ** 2).mean()
        total += w * emea
    return total / 255.0   # normalise


def _uiconm(image: np.ndarray, patch_size: int = 5) -> float:
    """
    Underwater Image Contrast Measure.
    AME (Amean × MEsigma) computed in patches.
    """
    grey      = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w      = grey.shape
    amee_vals = []
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = grey[i:i + patch_size, j:j + patch_size]
            mi, ma = patch.min(), patch.max()
            if ma > mi:
                amee = np.log((ma - mi) / (ma + mi + 1e-8))
                amee_vals.append(amee)
    return float(np.mean(amee_vals)) if amee_vals else 0.0


def compute_uiqm(
    image:    np.ndarray,
    c1: float = 0.0282,
    c2: float = 0.2953,
    c3: float = 3.5753,
) -> float:
    """
    Underwater Image Quality Measure (Panetta et al., 2016).
    Higher is better. Range: typically [-2, 3].

    UIQM = c1×UICM + c2×UISM + c3×UIConM

    Args:
        image: (H, W, 3) uint8 RGB
        c1, c2, c3: empirically determined coefficients from the paper
    """
    assert image.dtype == np.uint8, "Input must be uint8"
    uicm   = _uicm(image)
    uism   = _uism(image)
    uiconm = _uiconm(image)
    return c1 * uicm + c2 * uism + c3 * uiconm


# ════════════════════════════════════════════════════════════════
#  UCIQE — Underwater Color Image Quality Evaluation
# ════════════════════════════════════════════════════════════════

def compute_uciqe(image: np.ndarray) -> float:
    """
    Underwater Colour Image Quality Evaluation (Yang et al., 2015).
    No-reference metric. Higher is better. Range: [0, 1].

    UCIQE = c1×σ_c + c2×con_l + c3×μ_s

    where:
      σ_c   = chroma standard deviation (colour richness)
      con_l = luminance contrast (Michelson contrast)
      μ_s   = average saturation

    Args:
        image: (H, W, 3) uint8 RGB
    """
    assert image.dtype == np.uint8, "Input must be uint8"

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Chroma = sqrt(a² + b²)
    chroma     = np.sqrt(A ** 2 + B_ch ** 2)
    sigma_c    = chroma.std()

    # Luminance contrast (top 1% vs bottom 1%)
    l_flat = L.flatten()
    l_flat.sort()
    n      = len(l_flat)
    top    = l_flat[int(0.99 * n):].mean()
    bot    = l_flat[:int(0.01 * n) + 1].mean()
    con_l  = (top - bot) / (top + bot + 1e-8)

    # Saturation in HSV
    hsv   = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    mu_s  = hsv[:, :, 1].mean() / 255.0

    # Coefficients from the paper
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return float(c1 * sigma_c / 255.0 + c2 * con_l + c3 * mu_s)


# ════════════════════════════════════════════════════════════════
#  Batch evaluation
# ════════════════════════════════════════════════════════════════

def evaluate_batch(
    preds:   torch.Tensor,   # (B, 3, H, W) float32 [0, 1]
    targets: Optional[torch.Tensor] = None,  # None for no-reference eval
) -> Dict[str, float]:
    """
    Evaluate a batch of images.

    Returns:
        dict with available metrics.
    """
    results: Dict[str, float] = {}

    # Reference-based
    if targets is not None:
        results["psnr"] = compute_psnr(preds, targets)
        results["ssim"] = compute_ssim(preds, targets)

    # Convert to uint8 numpy for UIQM / UCIQE
    uiqm_scores  = []
    uciqe_scores = []

    preds_np = (preds.detach().cpu().clamp(0, 1) * 255).byte().numpy()
    # preds_np: (B, 3, H, W) uint8
    for i in range(preds_np.shape[0]):
        img = preds_np[i].transpose(1, 2, 0)   # (H, W, 3)
        uiqm_scores.append(compute_uiqm(img))
        uciqe_scores.append(compute_uciqe(img))

    results["uiqm"]  = float(np.mean(uiqm_scores))
    results["uciqe"] = float(np.mean(uciqe_scores))

    return results


# ════════════════════════════════════════════════════════════════
#  Full evaluation loop
# ════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Runs evaluation over a DataLoader and aggregates all metrics.
    """

    def __init__(
        self,
        model:  torch.nn.Module,
        device: torch.device,
    ):
        self.model  = model
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        loader:      torch.utils.data.DataLoader,
        save_dir:    Optional[str] = None,
        save_count:  int           = 20,
    ) -> Dict[str, float]:
        self.model.eval()

        psnr_list  = []
        ssim_list  = []
        uiqm_list  = []
        uciqe_list = []
        saved      = 0

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for batch in tqdm(loader, desc="Evaluating"):
            degraded  = batch["degraded"].to(self.device)
            has_ref   = "reference" in batch
            reference = batch["reference"].to(self.device) if has_ref else None

            out      = self.model(degraded)
            enhanced = out["enhanced"]

            metrics = evaluate_batch(enhanced, reference)

            if has_ref:
                psnr_list.append(metrics["psnr"])
                ssim_list.append(metrics["ssim"])
            uiqm_list.append(metrics["uiqm"])
            uciqe_list.append(metrics["uciqe"])

            # Save visuals
            if save_dir and saved < save_count:
                self._save_visual(
                    save_dir, saved,
                    degraded[0], enhanced[0],
                    reference[0] if has_ref else None,
                    out,
                )
                saved += 1

        summary = {
            "uiqm":  float(np.mean(uiqm_list)),
            "uciqe": float(np.mean(uciqe_list)),
        }
        if psnr_list:
            summary["psnr"] = float(np.mean(psnr_list))
            summary["ssim"] = float(np.mean(ssim_list))

        return summary

    def _save_visual(
        self,
        save_dir:  str,
        idx:       int,
        degraded:  torch.Tensor,
        enhanced:  torch.Tensor,
        reference: Optional[torch.Tensor],
        out:       dict,
    ):
        import matplotlib.pyplot as plt
        from ..utils.visualise import tensor_to_numpy

        n_cols = 5 if reference is not None else 4
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))

        def show(ax, t, title):
            img = tensor_to_numpy(t)
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        show(axes[0], degraded,               "Degraded Input")
        show(axes[1], enhanced,               "Enhanced Output")
        show(axes[2], out["noise_map"],        "Noise Map")
        show(axes[3], out["illum_map"],        "Illumination Map")
        if reference is not None:
            show(axes[4], reference,           "Reference (GT)")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/eval_{idx:04d}.png", dpi=150)
        plt.close()


# ─── missing import ───────────────────────────────────────────
from pathlib import Path
from tqdm import tqdm
