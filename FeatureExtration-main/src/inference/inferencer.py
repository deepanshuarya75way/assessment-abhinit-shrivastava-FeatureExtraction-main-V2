"""
src/inference/inferencer.py
============================
Inference Engine

Features:
  - Single image inference
  - Batch folder inference
  - Attention map visualisation
  - Illumination map visualisation
  - Side-by-side comparison output
  - Sliding-window tiled inference for very large images
"""

import os
import time
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
from tqdm import tqdm

from ..models.feature_extractor import UnderwaterFeatureExtractor


# ════════════════════════════════════════════════════════════════
#  Image I/O helpers
# ════════════════════════════════════════════════════════════════

def load_image(path: str, size: Optional[int] = None) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return img.astype(np.float32) / 255.0


def numpy_to_tensor(
    img: np.ndarray, device: torch.device
) -> torch.Tensor:
    """
    Convert (H, W, 3) float32 numpy → (1, 3, H, W) tensor on device.
    Applies ImageNet normalisation.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Convert (1, C, H, W) or (C, H, W) tensor → (H, W, C) uint8.
    Handles 1-channel (greyscale) and 3-channel tensors.
    Reverses ImageNet normalisation for 3-channel images.
    """
    if t.ndim == 4:
        t = t.squeeze(0)
    t = t.detach().cpu().float()

    if t.shape[0] == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t    = t * std + mean
        t    = t.permute(1, 2, 0).clamp(0, 1).numpy()
        return (t * 255).astype(np.uint8)
    else:
        # Single channel (noise / illumination map)
        t = t.squeeze(0).clamp(0, 1).numpy()
        return (t * 255).astype(np.uint8)


def save_image(img: np.ndarray, path: str):
    """Save (H, W, 3) uint8 RGB image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ════════════════════════════════════════════════════════════════
#  Colourmap helpers
# ════════════════════════════════════════════════════════════════

def apply_colormap(
    grey_map: np.ndarray, cmap: str = "jet"
) -> np.ndarray:
    """
    Apply a matplotlib colourmap to a single-channel map.

    Args:
        grey_map: (H, W) uint8 in [0, 255]
        cmap:     matplotlib colourmap name

    Returns:
        (H, W, 3) uint8 RGB
    """
    norm   = grey_map.astype(np.float32) / 255.0
    mapper = plt.get_cmap(cmap)
    rgb    = mapper(norm)[:, :, :3]
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(
    base: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Blend a coloured heatmap onto a base image (both uint8 RGB)."""
    assert base.shape[:2] == heatmap.shape[:2], "Size mismatch"
    return cv2.addWeighted(base, 1 - alpha, heatmap, alpha, 0)


# ════════════════════════════════════════════════════════════════
#  Core Inferencer
# ════════════════════════════════════════════════════════════════

class UnderwaterInferencer:
    """
    Inference engine for UnderwaterFeatureExtractor.

    Usage:
        inferencer = UnderwaterInferencer.from_checkpoint(
            checkpoint_path="checkpoints/best_model.pth",
            device="cuda",
        )
        result = inferencer.infer_single("my_image.jpg")
        inferencer.save_visualisation(result, "outputs/")
    """

    # ImageNet normalisation constants
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model:  UnderwaterFeatureExtractor,
        device: torch.device,
        image_size: int = 512,
    ):
        self.model      = model.to(device).eval()
        self.device     = device
        self.image_size = image_size

    # ──────────────────────────────────────────────────────────
    #  Constructors
    # ──────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device:          str  = "cuda",
        image_size:      int  = 512,
    ) -> "UnderwaterInferencer":
        """
        Load model from a saved checkpoint file.

        Args:
            checkpoint_path: path to .pth checkpoint
            device:          "cuda" | "cpu"
            image_size:      inference image size
        """
        dev   = torch.device(device if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint_path, map_location=dev)
        cfg   = state["config"]

        from ..models.feature_extractor import build_model
        model = build_model(cfg)
        model.load_state_dict(state["model"])
        print(f"Loaded checkpoint (epoch {state.get('epoch', '?')},"
              f" best PSNR={state.get('best_psnr', 0):.2f})")
        return cls(model, dev, image_size)

    # ──────────────────────────────────────────────────────────
    #  Single image inference
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer_single(
        self,
        image_path:    str,
        use_tta:       bool = False,   # test-time augmentation
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a single image.

        Args:
            image_path: path to raw underwater image
            use_tta:    average over horizontal + vertical flips

        Returns:
            dict with keys:
              "original"     : (H, W, 3) uint8 original image
              "enhanced"     : (H, W, 3) uint8 enhanced image
              "noise_map"    : (H, W, 3) uint8 coloured noise map
              "illum_map"    : (H, W, 3) uint8 coloured illumination map
              "colour_map"   : (H, W, 3) uint8 colour deviation (normalised)
              "attn_stage1"  : (H, W, 3) uint8 stage-1 attention heatmap
              "attn_stage2"  : (H, W, 3) uint8 stage-2 attention heatmap
              "attn_stage3"  : (H, W, 3) uint8 stage-3 attention heatmap
              "overlay_noise": (H, W, 3) uint8 noise heatmap overlaid on original
        """
        orig_img  = load_image(image_path, size=self.image_size)
        orig_uint = (orig_img * 255).astype(np.uint8)

        tensor  = numpy_to_tensor(orig_img, self.device)

        if use_tta:
            out = self._tta_forward(tensor)
        else:
            out = self.model(tensor)

        return self._pack_outputs(orig_uint, out)

    @torch.no_grad()
    def infer_tensor(
        self, tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Direct inference on a (B, 3, H, W) tensor already on device.
        Returns raw model output dict.
        """
        return self.model(tensor)

    # ──────────────────────────────────────────────────────────
    #  Batch folder inference
    # ──────────────────────────────────────────────────────────

    def infer_folder(
        self,
        input_dir:  str,
        output_dir: str,
        visualise:  bool = True,
    ) -> List[Dict]:
        """
        Process all images in a folder.

        Args:
            input_dir:  folder with raw underwater images
            output_dir: where to save enhanced images + visuals
            visualise:  also save full visualisation panels
        """
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "enhanced").mkdir(exist_ok=True)
        if visualise:
            (output_dir / "visuals").mkdir(exist_ok=True)

        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)
        assert len(files) > 0, f"No images found in {input_dir}"

        results = []
        times   = []
        for path in tqdm(files, desc="Inference"):
            t0     = time.time()
            result = self.infer_single(str(path))
            elapsed = time.time() - t0
            times.append(elapsed)

            # Save enhanced image
            save_image(
                result["enhanced"],
                str(output_dir / "enhanced" / (path.stem + "_enhanced.png")),
            )

            if visualise:
                self.save_visualisation(
                    result,
                    str(output_dir / "visuals" / (path.stem + "_visual.png")),
                )

            results.append({"path": str(path), "time": elapsed, **result})

        avg_time = np.mean(times)
        fps      = 1.0 / avg_time
        print(f"\nProcessed {len(files)} images")
        print(f"Avg time per image: {avg_time*1000:.1f} ms  ({fps:.1f} FPS)")
        return results

    # ──────────────────────────────────────────────────────────
    #  Tiled inference for high-resolution images
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer_tiled(
        self,
        image_path: str,
        tile_size:  int = 512,
        overlap:    int = 64,
    ) -> np.ndarray:
        """
        Process a very large image using overlapping tiles.
        Blends tiles with Gaussian weighting to avoid seam artefacts.

        Returns:
            enhanced: (H, W, 3) uint8 enhanced image at original resolution
        """
        orig = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W = orig.shape[:2]

        output  = np.zeros_like(orig)
        weights = np.zeros((H, W, 1), dtype=np.float32)

        # Gaussian blend window
        step = tile_size - overlap
        gwin_1d = np.hanning(tile_size).astype(np.float32)
        gwin    = np.outer(gwin_1d, gwin_1d)[:, :, np.newaxis]

        y_starts = list(range(0, H - tile_size + 1, step)) + [H - tile_size]
        x_starts = list(range(0, W - tile_size + 1, step)) + [W - tile_size]

        for y in y_starts:
            for x in x_starts:
                tile = orig[y:y + tile_size, x:x + tile_size]
                t    = numpy_to_tensor(tile, self.device)
                out  = self.model(t)
                enh  = tensor_to_numpy(out["enhanced"]).astype(np.float32) / 255.0
                output[y:y + tile_size, x:x + tile_size]  += enh * gwin
                weights[y:y + tile_size, x:x + tile_size] += gwin

        enhanced = (output / (weights + 1e-8)).clip(0, 1)
        return (enhanced * 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────
    #  TTA (Test-Time Augmentation)
    # ──────────────────────────────────────────────────────────

    def _tta_forward(
        self, tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Average predictions over 4 flips for more robust output."""
        augmentations = [
            lambda x: x,
            lambda x: torch.flip(x, dims=[-1]),         # H-flip
            lambda x: torch.flip(x, dims=[-2]),         # V-flip
            lambda x: torch.flip(x, dims=[-1, -2]),     # both
        ]
        de_augmentations = [
            lambda x: x,
            lambda x: torch.flip(x, dims=[-1]),
            lambda x: torch.flip(x, dims=[-2]),
            lambda x: torch.flip(x, dims=[-1, -2]),
        ]

        enhanced_sum  = None
        noise_sum     = None
        illum_sum     = None

        for aug, de_aug in zip(augmentations, de_augmentations):
            out = self.model(aug(tensor))
            enh = de_aug(out["enhanced"])
            nse = de_aug(out["noise_map"])
            ill = de_aug(out["illum_map"])

            enhanced_sum = enh if enhanced_sum is None else enhanced_sum + enh
            noise_sum    = nse if noise_sum    is None else noise_sum    + nse
            illum_sum    = ill if illum_sum    is None else illum_sum    + ill

        n = len(augmentations)
        out["enhanced"]  = enhanced_sum / n
        out["noise_map"] = noise_sum    / n
        out["illum_map"] = illum_sum    / n
        return out

    # ──────────────────────────────────────────────────────────
    #  Pack raw model outputs → visual numpy arrays
    # ──────────────────────────────────────────────────────────

    def _pack_outputs(
        self,
        orig_uint: np.ndarray,
        out:       Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        H, W = orig_uint.shape[:2]

        # Enhanced image
        enh_uint = tensor_to_numpy(out["enhanced"])
        enh_uint = cv2.resize(enh_uint, (W, H), interpolation=cv2.INTER_LANCZOS4)

        # Noise map (grayscale → coloured)
        noise_grey = tensor_to_numpy(out["noise_map"])   # (H, W) uint8
        noise_col  = apply_colormap(noise_grey, "hot")
        noise_col  = cv2.resize(noise_col, (W, H))

        # Illumination map
        illum_grey = tensor_to_numpy(out["illum_map"])
        illum_col  = apply_colormap(illum_grey, "inferno")
        illum_col  = cv2.resize(illum_col, (W, H))

        # Colour deviation map (3-channel, Tanh output → shift to [0,1])
        cmap_t      = out["colour_map"].squeeze(0).detach().cpu()
        cmap_t      = (cmap_t + 1.0) / 2.0             # [-1,1] → [0,1]
        cmap_uint   = (cmap_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cmap_uint   = cv2.resize(cmap_uint, (W, H))

        # Stage attention maps (mean over channels → heatmap)
        stage_attn = {}
        for i, sf in enumerate(out["stage_feats"]):
            mean_map = sf.mean(dim=1, keepdim=True)     # (1, 1, h, w)
            grey     = tensor_to_numpy(mean_map)
            col_map  = apply_colormap(grey, "viridis")
            col_map  = cv2.resize(col_map, (W, H))
            stage_attn[f"attn_stage{i+1}"] = col_map

        # Noise overlay on original
        noise_overlay = overlay_heatmap(orig_uint.copy(), noise_col, alpha=0.45)

        result = {
            "original":      orig_uint,
            "enhanced":      enh_uint,
            "noise_map":     noise_col,
            "illum_map":     illum_col,
            "colour_map":    cmap_uint,
            "overlay_noise": noise_overlay,
        }
        result.update(stage_attn)
        return result

    # ──────────────────────────────────────────────────────────
    #  Visualisation panel
    # ──────────────────────────────────────────────────────────

    def save_visualisation(
        self,
        result:    Dict[str, np.ndarray],
        save_path: str,
        dpi:       int = 150,
    ):
        """
        Save a comprehensive 3×3 visualisation panel:

          [Original]  [Enhanced]      [Noise Map]
          [Illum Map] [Colour Deviation] [Noise Overlay]
          [Attn S1]   [Attn S2]       [Attn S3]
        """
        keys = [
            ("original",     "Original Input"),
            ("enhanced",     "Enhanced Output"),
            ("noise_map",    "Noise / Backscatter Map"),
            ("illum_map",    "Illumination Map"),
            ("colour_map",   "Colour Deviation Map"),
            ("overlay_noise","Noise Overlay"),
            ("attn_stage1",  "Attention — Stage 1"),
            ("attn_stage2",  "Attention — Stage 2"),
            ("attn_stage3",  "Attention — Stage 3"),
        ]

        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor("#1a1a2e")

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.12, wspace=0.05)

        for idx, (key, title) in enumerate(keys):
            ax  = fig.add_subplot(gs[idx // 3, idx % 3])
            img = result.get(key)
            if img is None:
                ax.axis("off")
                continue

            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)

            ax.set_title(title, fontsize=10, color="white", pad=6)
            ax.axis("off")

            # Red border on "enhanced" to highlight it
            if key == "enhanced":
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("#e94560")
                    spine.set_linewidth(2.5)

        plt.suptitle(
            "Underwater Feature Extraction — Inference Visualisation",
            fontsize=14, color="white", y=1.01, fontweight="bold",
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved visualisation → {save_path}")

    # ──────────────────────────────────────────────────────────
    #  Quick comparison: degraded vs enhanced side by side
    # ──────────────────────────────────────────────────────────

    def save_comparison(
        self,
        result:    Dict[str, np.ndarray],
        save_path: str,
    ):
        """Minimal side-by-side: original | enhanced."""
        orig = result["original"]
        enh  = result["enhanced"]
        canvas = np.concatenate([orig, enh], axis=1)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(canvas)
        ax.axvline(orig.shape[1], color="white", linewidth=2, linestyle="--")
        ax.set_title("Before (left) vs After (right)", fontsize=13, pad=8)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
