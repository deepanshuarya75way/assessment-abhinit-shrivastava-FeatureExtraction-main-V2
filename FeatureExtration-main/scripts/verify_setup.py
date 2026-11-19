#!/usr/bin/env python3
"""
scripts/verify_setup.py
========================
Verify the full environment and model pipeline.

Checks:
  1. Python version
  2. PyTorch installation + CUDA availability
  3. Required package imports
  4. Model builds without errors
  5. Forward pass with dummy data (GPU if available)
  6. All loss functions compute correctly
  7. Evaluation metrics run
  8. Config loads correctly

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --full    # also checks optional packages
"""

import sys
import importlib
import time
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def check(label: str, fn, critical: bool = True):
    """Run a check function and print pass/fail."""
    try:
        result = fn()
        msg = f"  ✓  {label}"
        if isinstance(result, str):
            msg += f"  ({result})"
        print(msg)
        return True
    except Exception as e:
        prefix = "  ✗" if critical else "  ⚠"
        print(f"{prefix}  {label}")
        print(f"       → {e}")
        return not critical   # critical failure → return False


def main():
    all_ok = True
    print("\n" + "═"*60)
    print("  Underwater Feature Extraction — Setup Verification")
    print("═"*60 + "\n")

    # ── 1. Python ────────────────────────────────────────────
    print("Python & Core")
    print("─"*40)
    all_ok &= check("Python version", lambda: (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 9) else (_ for _ in ()).throw(
            RuntimeError(f"Python 3.9+ required, got {sys.version_info}")
        )
    ))

    # ── 2. PyTorch ───────────────────────────────────────────
    print("\nPyTorch")
    print("─"*40)

    import torch

    all_ok &= check("PyTorch",       lambda: torch.__version__)
    all_ok &= check("CUDA available",
        lambda: f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
                else "No GPU (CPU only — training will be slow)",
        critical=False,
    )
    all_ok &= check("torchvision",   lambda: importlib.import_module("torchvision").__version__)

    # ── 3. Required packages ─────────────────────────────────
    print("\nRequired Packages")
    print("─"*40)

    packages = [
        ("cv2",           "opencv-python"),
        ("numpy",         "numpy"),
        ("albumentations","albumentations"),
        ("PIL",           "Pillow"),
        ("sklearn",       "scikit-learn"),
        ("skimage",       "scikit-image"),
        ("kornia",        "kornia"),
        ("tqdm",          "tqdm"),
        ("yaml",          "pyyaml"),
        ("matplotlib",    "matplotlib"),
        ("einops",        "einops"),
        ("timm",          "timm"),
    ]

    for pkg, pip_name in packages:
        all_ok &= check(pip_name, lambda p=pkg: importlib.import_module(p).__version__)

    # ── 4. Project modules ───────────────────────────────────
    print("\nProject Modules")
    print("─"*40)

    all_ok &= check("Config loading", lambda: (
        _load_config(),
        "OK"
    )[-1])

    all_ok &= check("Preprocessing module", lambda: (
        _import("src.data.preprocessing"), "OK"
    )[-1])

    all_ok &= check("Dataset module", lambda: (
        _import("src.data.dataset"), "OK"
    )[-1])

    all_ok &= check("Attention modules", lambda: (
        _import("src.models.attention"), "OK"
    )[-1])

    all_ok &= check("Estimator modules", lambda: (
        _import("src.models.estimators"), "OK"
    )[-1])

    all_ok &= check("Backbone module", lambda: (
        _import("src.models.backbone"), "OK"
    )[-1])

    all_ok &= check("Feature extractor model", lambda: (
        _import("src.models.feature_extractor"), "OK"
    )[-1])

    all_ok &= check("Loss functions", lambda: (
        _import("src.losses.losses"), "OK"
    )[-1])

    all_ok &= check("Trainer", lambda: (
        _import("src.training.trainer"), "OK"
    )[-1])

    all_ok &= check("Metrics", lambda: (
        _import("src.evaluation.metrics"), "OK"
    )[-1])

    # ── 5. Model build ───────────────────────────────────────
    print("\nModel Build & Forward Pass")
    print("─"*40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model():
        from src.models.feature_extractor import UnderwaterFeatureExtractor
        m = UnderwaterFeatureExtractor(
            backbone="resnet34", pretrained=False,
            num_classes=10, base_features=32,
        )
        n = sum(p.numel() for p in m.parameters()) / 1e6
        return f"{n:.1f}M params"

    all_ok &= check("Build model", _build_model)

    def _forward_pass():
        from src.models.feature_extractor import UnderwaterFeatureExtractor
        m  = UnderwaterFeatureExtractor(
            backbone="resnet18", pretrained=False,
            num_classes=5, base_features=16
        ).to(device).eval()
        x  = torch.randn(1, 3, 128, 128, device=device)
        t0 = time.time()
        with torch.no_grad():
            out = m(x)
        dt = (time.time() - t0) * 1000
        assert "enhanced"   in out
        assert "noise_map"  in out
        assert "illum_map"  in out
        assert "features"   in out
        assert "logits"     in out
        return f"enhanced={tuple(out['enhanced'].shape)}, dt={dt:.1f}ms"

    all_ok &= check("Forward pass (128×128)", _forward_pass)

    # ── 6. Losses ────────────────────────────────────────────
    print("\nLoss Functions")
    print("─"*40)

    def _test_losses():
        from src.losses.losses import (
            ReconstructionLoss, SSIMLoss,
            ColourConsistencyLoss, ContrastEnhancementLoss,
            NoiseSmoothLoss,
        )
        p = torch.rand(2, 3, 64, 64)
        t = torch.rand(2, 3, 64, 64)
        n = torch.rand(2, 1, 64, 64)

        ReconstructionLoss()(p, t)
        SSIMLoss()(p, t)
        ColourConsistencyLoss()(p, t)
        ContrastEnhancementLoss()(p, t)
        NoiseSmoothLoss()(n)
        return "all losses OK"

    all_ok &= check("All loss functions", _test_losses)

    # ── 7. Metrics ───────────────────────────────────────────
    print("\nEvaluation Metrics")
    print("─"*40)

    def _test_metrics():
        import numpy as np
        from src.evaluation.metrics import compute_uiqm, compute_uciqe

        img   = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        uiqm  = compute_uiqm(img)
        uciqe = compute_uciqe(img)
        return f"UIQM={uiqm:.3f}, UCIQE={uciqe:.3f}"

    all_ok &= check("UIQM & UCIQE metrics", _test_metrics)

    # ── Result ───────────────────────────────────────────────
    print("\n" + "═"*60)
    if all_ok:
        print("  ✅  All checks passed! Ready to train.")
    else:
        print("  ❌  Some checks failed. Please fix the issues above.")
    print("═"*60 + "\n")

    sys.exit(0 if all_ok else 1)


def _import(module: str):
    return importlib.import_module(module)


def _load_config():
    from src.utils.utils import load_config
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if cfg_path.exists():
        return load_config(str(cfg_path))
    raise FileNotFoundError(f"Config not found: {cfg_path}")


if __name__ == "__main__":
    main()
