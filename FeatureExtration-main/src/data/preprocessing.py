"""
src/data/preprocessing.py
==========================
Domain-Specific Underwater Image Preprocessing

Addresses:
  1. Color cast  — blue / green dominance from selective light absorption
  2. White balance — Gray-World & learned correction
  3. Backscatter — high-frequency noise + haze-like veil
  4. Light attenuation with depth — non-uniform illumination
  5. Histogram-based contrast stretching

All functions operate on:
  - NumPy arrays  (H, W, 3)  uint8 or float32 in [0, 1]
  - Torch tensors (C, H, W)  float32 (for differentiable versions)
"""

from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.color import rgb_to_lab, lab_to_rgb


# ════════════════════════════════════════════════════════════════
#  1. Color Cast Correction
# ════════════════════════════════════════════════════════════════

def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Classic Gray-World assumption: average colour should be neutral grey.
    Works well for images where scene colours are diverse.

    Args:
        image: (H, W, 3) uint8 BGR or RGB

    Returns:
        corrected: (H, W, 3) uint8
    """
    img = image.astype(np.float32)
    mean = img.mean(axis=(0, 1))          # (3,)
    mean_grey = mean.mean()               # scalar
    scale = mean_grey / (mean + 1e-8)
    corrected = np.clip(img * scale, 0, 255).astype(np.uint8)
    return corrected


def stretch_channel(channel: np.ndarray) -> np.ndarray:
    """Linear stretch of a single channel to [0, 255]."""
    lo, hi = channel.min(), channel.max()
    if hi == lo:
        return channel
    return ((channel - lo) / (hi - lo) * 255).astype(np.uint8)


def histogram_stretch_rgb(image: np.ndarray) -> np.ndarray:
    """
    Per-channel histogram stretching — compensates for colour attenuation.
    Red channel is most attenuated underwater, so this aggressively restores it.

    Args:
        image: (H, W, 3) uint8 RGB

    Returns:
        stretched: (H, W, 3) uint8 RGB
    """
    stretched = np.stack(
        [stretch_channel(image[:, :, c]) for c in range(3)], axis=-1
    )
    return stretched


def clahe_per_channel(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalisation on each channel.
    Better than global HE — preserves local contrast without over-amplifying noise.

    Args:
        image: (H, W, 3) uint8 RGB

    Returns:
        equalized: (H, W, 3) uint8 RGB
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    channels = cv2.split(image)
    enhanced = [clahe.apply(c) for c in channels]
    return cv2.merge(enhanced)


def underwater_color_correction(image: np.ndarray) -> np.ndarray:
    """
    Composite color correction tailored for underwater images:
      1. Histogram stretch (restore lost red channel)
      2. Gray-world white balance
      3. CLAHE (boost local contrast)

    Args:
        image: (H, W, 3) uint8 RGB

    Returns:
        corrected: (H, W, 3) uint8 RGB
    """
    img = histogram_stretch_rgb(image)
    img = gray_world_white_balance(img)
    img = clahe_per_channel(img)
    return img


# ════════════════════════════════════════════════════════════════
#  2. Backscatter Estimation & Removal
# ════════════════════════════════════════════════════════════════

def estimate_backscatter_map(image: np.ndarray, radius: int = 21) -> np.ndarray:
    """
    Estimate backscatter veil using a large-kernel Gaussian blur.
    Backscatter appears as a low-frequency haze overlaid on the scene.

    Based on: "Sea-thru: A Method for Removing Water From Underwater Images"
    (Akkaynak & Treibitz, CVPR 2019)

    Args:
        image:  (H, W, 3) float32 in [0, 1]
        radius: kernel radius (larger = more aggressive backscatter estimate)

    Returns:
        backscatter: (H, W, 3) float32 — estimated backscatter component
    """
    ksize = 2 * radius + 1
    backscatter = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=radius / 2.0)
    # Clip so backscatter ≤ image
    backscatter = np.minimum(backscatter, image)
    return backscatter


def remove_backscatter(image: np.ndarray, radius: int = 21) -> np.ndarray:
    """
    Subtract estimated backscatter and renormalize.

    Args:
        image:  (H, W, 3) float32 [0, 1]

    Returns:
        dehazesd: (H, W, 3) float32 [0, 1]
    """
    backscatter = estimate_backscatter_map(image, radius=radius)
    dehazed = image - backscatter
    # Normalize to [0, 1]
    lo, hi = dehazed.min(), dehazed.max()
    if hi > lo:
        dehazed = (dehazed - lo) / (hi - lo)
    return np.clip(dehazed, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════
#  3. Illumination Map Estimation (classical, for preprocessing)
# ════════════════════════════════════════════════════════════════

def estimate_illumination_map(
    image: np.ndarray,
    sigma: float = 15.0,
) -> np.ndarray:
    """
    Approximate illumination map via max-channel + Gaussian smoothing.
    This classical approach seeds the deep network's illumination branch.

    Args:
        image:  (H, W, 3) float32 [0, 1]
        sigma:  Gaussian std (larger = smoother illumination estimate)

    Returns:
        illum:  (H, W) float32  — illumination map in [0, 1]
    """
    max_channel = image.max(axis=-1)                     # (H, W)
    illum = cv2.GaussianBlur(max_channel, (0, 0), sigma)
    return np.clip(illum, 0.0, 1.0)


def correct_illumination(
    image: np.ndarray,
    illum: Optional[np.ndarray] = None,
    target_brightness: float = 0.5,
) -> np.ndarray:
    """
    Divide by illumination map to get a 'flat field' corrected image.

    Args:
        image:              (H, W, 3) float32
        illum:              (H, W)   float32 — if None, estimated internally
        target_brightness:  desired mean brightness after correction

    Returns:
        corrected: (H, W, 3) float32 [0, 1]
    """
    if illum is None:
        illum = estimate_illumination_map(image)

    illum_3c = illum[:, :, np.newaxis]           # broadcast to 3 channels
    corrected = image / (illum_3c + 1e-8)

    # Rescale so mean = target_brightness
    current_mean = corrected.mean()
    if current_mean > 0:
        corrected = corrected * (target_brightness / current_mean)

    return np.clip(corrected, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════
#  4. Full Preprocessing Pipeline
# ════════════════════════════════════════════════════════════════

class UnderwaterPreprocessor:
    """
    Stateless preprocessing pipeline for underwater images.

    Designed to be called:
      - Offline during dataset preparation
      - Online (inside Dataset.__getitem__) for fast augmentation

    Steps (in order):
        1. Resize to target size
        2. Color cast correction (histogram stretch → white balance → CLAHE)
        3. Backscatter removal
        4. Illumination correction
        5. Convert to float32 [0, 1]
    """

    def __init__(
        self,
        image_size: int = 256,
        apply_color_correction: bool = True,
        apply_backscatter_removal: bool = True,
        apply_illumination_correction: bool = True,
        backscatter_radius: int = 21,
    ):
        self.image_size                   = image_size
        self.apply_color_correction       = apply_color_correction
        self.apply_backscatter_removal    = apply_backscatter_removal
        self.apply_illumination_correction = apply_illumination_correction
        self.backscatter_radius           = backscatter_radius

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: (H, W, 3) uint8 RGB

        Returns:
            processed: (H, W, 3) float32 [0, 1]
        """
        # 1. Resize
        h, w = image.shape[:2]
        if h != self.image_size or w != self.image_size:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # 2. Color correction (on uint8)
        if self.apply_color_correction:
            image = underwater_color_correction(image)

        # 3. Convert to float32 [0, 1]
        img_f = image.astype(np.float32) / 255.0

        # 4. Backscatter removal
        if self.apply_backscatter_removal:
            img_f = remove_backscatter(img_f, radius=self.backscatter_radius)

        # 5. Illumination correction
        if self.apply_illumination_correction:
            img_f = correct_illumination(img_f)

        return img_f

    def preprocess_pair(
        self,
        degraded: np.ndarray,
        reference: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess both images in a paired sample.
        Geometric transform must have been applied beforehand.
        """
        return self.preprocess_degraded(degraded), self.preprocess_reference(reference)

    def preprocess_degraded(self, image: np.ndarray) -> np.ndarray:
        """Full pipeline for degraded image."""
        return self(image)

    def preprocess_reference(self, image: np.ndarray) -> np.ndarray:
        """Minimal preprocessing for reference — only resize + normalize."""
        h, w = image.shape[:2]
        if h != self.image_size or w != self.image_size:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return image.astype(np.float32) / 255.0


# ════════════════════════════════════════════════════════════════
#  5. Differentiable utilities (for use inside model / loss)
# ════════════════════════════════════════════════════════════════

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB [0,1] tensor (B, 3, H, W) to YCbCr.
    Useful for computing colour-aware losses.
    """
    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    Y  =  0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.1687 * r - 0.3313 * g + 0.500 * b + 0.5
    Cr =  0.500 * r - 0.4187 * g - 0.0813 * b + 0.5
    return torch.stack([Y, Cb, Cr], dim=1)


def compute_laplacian_map(image: torch.Tensor) -> torch.Tensor:
    """
    Compute a sharpness / edge map using Laplacian operator.
    High values → sharp regions; low values → blurred / hazy regions.

    Args:
        image: (B, 3, H, W) float32

    Returns:
        edge_map: (B, 1, H, W) float32
    """
    # Laplacian kernel
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32,
        device=image.device,
    ).view(1, 1, 3, 3)

    # Convert to greyscale
    grey = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    lap  = F.conv2d(grey, kernel, padding=1)
    return lap.abs()


def compute_dark_channel(
    image: torch.Tensor, patch_size: int = 15
) -> torch.Tensor:
    """
    Dark Channel Prior (He et al.) — adapted for underwater haze estimation.

    Args:
        image:      (B, 3, H, W) float32 [0, 1]
        patch_size: local minimum patch size

    Returns:
        dark_channel: (B, 1, H, W) float32
    """
    # Min across colour channels
    min_channel, _ = image.min(dim=1, keepdim=True)   # (B, 1, H, W)
    # Min-pool to get local minimum
    padding = patch_size // 2
    dark = -F.max_pool2d(
        -min_channel, kernel_size=patch_size, stride=1, padding=padding
    )
    return dark
