"""
src/utils/visualise.py
=======================
Visualisation Utilities

Exports:
  - tensor_to_numpy          — safely convert any model output tensor to numpy
  - visualise_feature_maps   — grid plot of CNN feature maps
  - visualise_attention_evolution — how attention develops across stages
  - plot_metrics_history     — plot PSNR/SSIM/UIQM training curves
  - make_comparison_grid     — side-by-side grid of multiple images
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ════════════════════════════════════════════════════════════════
#  Tensor → Numpy conversion
# ════════════════════════════════════════════════════════════════

def tensor_to_numpy(
    t: torch.Tensor,
    denormalize: bool = True,
) -> np.ndarray:
    """
    Convert a (C, H, W) or (1, C, H, W) or (B, C, H, W) tensor to
    (H, W, C) uint8 numpy array for display.

    Args:
        t:           input tensor (any shape mentioned above)
        denormalize: if True + 3 channels, reverse ImageNet normalisation

    Returns:
        (H, W, C) uint8  or  (H, W) uint8 for single-channel
    """
    if t.ndim == 4:
        t = t[0]   # take first batch item
    t = t.detach().cpu().float()

    if t.shape[0] == 3 and denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t    = t * std + mean
        img  = t.permute(1, 2, 0).clamp(0, 1).numpy()
        return (img * 255).astype(np.uint8)
    elif t.shape[0] == 1:
        img = t.squeeze(0).clamp(0, 1).numpy()
        return (img * 255).astype(np.uint8)
    elif t.shape[0] == 3 and not denormalize:
        img = t.permute(1, 2, 0).clamp(0, 1).numpy()
        return (img * 255).astype(np.uint8)
    else:
        # Multi-channel feature map → average across channels
        img = t.mean(0).clamp(0, 1).numpy()
        # Normalise for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img * 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════
#  Feature Map Grid
# ════════════════════════════════════════════════════════════════

def visualise_feature_maps(
    feature_map: torch.Tensor,
    max_channels: int  = 16,
    save_path:    Optional[str] = None,
    title:        str  = "Feature Maps",
) -> np.ndarray:
    """
    Display a grid of individual CNN feature map channels.

    Args:
        feature_map:  (B, C, H, W) or (C, H, W) tensor
        max_channels: maximum number of channels to display
        save_path:    if provided, save figure here
        title:        plot title

    Returns:
        (H, W, 3) uint8 numpy image of the grid
    """
    if feature_map.ndim == 4:
        fm = feature_map[0]
    else:
        fm = feature_map
    fm = fm.detach().cpu().float()

    n_channels = min(fm.shape[0], max_channels)
    n_cols     = min(8, n_channels)
    n_rows     = (n_channels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    fig.patch.set_facecolor("#1a1a2e")
    plt.suptitle(title, color="white", fontsize=12, y=1.01)

    # Flatten axes for indexing
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(len(axes)):
        ax = axes[i]
        if i < n_channels:
            ch = fm[i].numpy()
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            ax.imshow(ch, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"ch{i}", fontsize=7, color="white")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    # Convert to numpy
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return buf


# ════════════════════════════════════════════════════════════════
#  Attention Evolution Across Stages
# ════════════════════════════════════════════════════════════════

def visualise_attention_evolution(
    original:    np.ndarray,                 # (H, W, 3) uint8
    stage_feats: List[torch.Tensor],         # list of (1, C, h, w) tensors
    save_path:   Optional[str] = None,
) -> np.ndarray:
    """
    Show how attention evolves from early to deep network stages.
    Row 1: original at each scale
    Row 2: attention heatmap at each stage

    Args:
        original:    original input image
        stage_feats: list of feature tensors from each attention stage
    """
    n = len(stage_feats)
    H, W = original.shape[:2]

    fig, axes = plt.subplots(2, n + 1, figsize=((n + 1) * 3.5, 7))
    fig.patch.set_facecolor("#0f0f1a")

    # Top row: input at different scales
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Input", color="white", fontsize=10)
    axes[0, 0].axis("off")

    cmaps = ["Reds", "Oranges", "YlOrRd"]
    for i, sf in enumerate(stage_feats):
        scale = 2 ** (i + 2)   # stages at /4, /8, /16
        small = cv2.resize(original, (W // scale, H // scale))
        axes[0, i + 1].imshow(small)
        axes[0, i + 1].set_title(f"Input ×1/{scale}", color="white", fontsize=9)
        axes[0, i + 1].axis("off")

    # Bottom row: attention heatmaps
    axes[1, 0].axis("off")   # blank under original
    for i, sf in enumerate(stage_feats):
        am = sf.mean(dim=1, keepdim=True)      # (1, 1, h, w)
        am = F.interpolate(am, size=(H, W), mode="bilinear", align_corners=False)
        am = am.squeeze().detach().cpu().numpy()
        am = (am - am.min()) / (am.max() - am.min() + 1e-8)

        axes[1, i + 1].imshow(original)
        axes[1, i + 1].imshow(am, cmap=cmaps[i % len(cmaps)], alpha=0.55, vmin=0, vmax=1)
        axes[1, i + 1].set_title(f"Attention Stage {i+1}", color="white", fontsize=9)
        axes[1, i + 1].axis("off")

    plt.suptitle("Attention Map Evolution Across Network Stages",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout(pad=0.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return buf


# ════════════════════════════════════════════════════════════════
#  Training Metrics Plot
# ════════════════════════════════════════════════════════════════

def plot_metrics_history(
    history:   Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Plot training metric history from a dict of metric name → list of values.

    Args:
        history:   e.g. {"psnr": [...], "ssim": [...], "uiqm": [...]}
        save_path: where to save the figure

    Returns:
        (H, W, 3) uint8 figure as numpy array
    """
    keys   = list(history.keys())
    n_rows = (len(keys) + 1) // 2
    n_cols = min(2, len(keys))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    fig.patch.set_facecolor("#1a1a2e")

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = list(axes)

    flat_axes = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    colours = ["#e94560", "#0f3460", "#533483", "#16213e", "#1a1a2e"]

    for i, key in enumerate(keys):
        ax     = flat_axes[i]
        values = history[key]
        epochs = list(range(1, len(values) + 1))

        ax.set_facecolor("#16213e")
        ax.plot(epochs, values, color=colours[i % len(colours)],
                linewidth=2.0, label=key)
        ax.fill_between(epochs, values, alpha=0.15,
                        color=colours[i % len(colours)])
        ax.set_title(key.upper(), color="white", fontsize=11)
        ax.tick_params(colors="grey")
        ax.spines[:].set_color("#333355")
        ax.grid(alpha=0.2, color="grey")
        ax.legend(fontsize=9, labelcolor="white", facecolor="#0f0f1a")

    # Hide unused axes
    for j in range(len(keys), len(flat_axes)):
        flat_axes[j].axis("off")

    plt.suptitle("Training Metrics History", color="white", fontsize=13, y=1.01)
    plt.tight_layout(pad=1.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return buf


# ════════════════════════════════════════════════════════════════
#  Comparison Grid (multi-image)
# ════════════════════════════════════════════════════════════════

def make_comparison_grid(
    images:    List[np.ndarray],   # each (H, W, 3) uint8
    labels:    List[str],
    n_cols:    int          = 4,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Make a labelled grid of images for paper/portfolio figures.

    Args:
        images:    list of (H, W, 3) uint8 images (all same size)
        labels:    label for each image
        n_cols:    images per row
        save_path: optional save path

    Returns:
        (H, W, 3) uint8 grid image
    """
    assert len(images) == len(labels)
    n_rows = (len(images) + n_cols - 1) // n_cols
    H, W   = images[0].shape[:2]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    fig.patch.set_facecolor("#0d0d1a")

    if n_rows == 1:
        axes_flat = [axes] if n_cols == 1 else list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for i, (img, label) in enumerate(zip(images, labels)):
        ax = axes_flat[i]
        ax.imshow(img)
        ax.set_title(label, color="white", fontsize=9, pad=4)
        ax.axis("off")

    for j in range(len(images), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout(pad=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return buf
