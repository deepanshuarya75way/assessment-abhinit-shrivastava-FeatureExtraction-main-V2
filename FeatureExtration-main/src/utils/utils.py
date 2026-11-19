"""
src/utils/utils.py
===================
Common Utilities

- Config loading (YAML)
- Reproducibility seeding
- Rich console logger
- Model parameter summary
- Checkpoint management
- Timing context manager
"""

import os
import random
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml


# ════════════════════════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════════════════════════

def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file.

    Args:
        path: path to .yaml / .yml file

    Returns:
        config dict
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(config: Dict, path: str):
    """Dump config dict to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two config dicts.
    Values in `override` take precedence.
    """
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


# ════════════════════════════════════════════════════════════════
#  Reproducibility
# ════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    Covers Python, NumPy, PyTorch (CPU + GPU), and CuDNN determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False  # Slower but deterministic


def get_device(device_str: str = "cuda") -> torch.device:
    """
    Resolve device string to a torch.device.
    Falls back gracefully if CUDA is unavailable.
    """
    if device_str == "cuda":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            warnings.warn("CUDA not available, falling back to CPU.")
            dev = torch.device("cpu")
    elif device_str == "mps":
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
            print("Using Apple MPS")
        else:
            warnings.warn("MPS not available, falling back to CPU.")
            dev = torch.device("cpu")
    else:
        dev = torch.device("cpu")
        print("Using CPU")
    return dev


# ════════════════════════════════════════════════════════════════
#  Model Summary
# ════════════════════════════════════════════════════════════════

def model_summary(model: nn.Module, input_size: tuple = (1, 3, 256, 256)):
    """
    Print a concise model summary.

    Args:
        model:      PyTorch model
        input_size: (B, C, H, W) example input size
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print("\n" + "═" * 55)
    print("  Model Summary")
    print("═" * 55)
    print(f"  {'Total parameters:':<30} {total:>12,}")
    print(f"  {'Trainable parameters:':<30} {trainable:>12,}")
    print(f"  {'Frozen parameters:':<30} {frozen:>12,}")
    print(f"  {'Memory (fp32 estimate):':<30} {total * 4 / 1e6:>11.1f} MB")

    # Per-module breakdown
    print("\n  Sub-module breakdown:")
    print("  " + "-" * 53)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<28} {params:>12,}  params")
    print("═" * 55 + "\n")


# ════════════════════════════════════════════════════════════════
#  Timing
# ════════════════════════════════════════════════════════════════

@contextmanager
def timer(label: str = ""):
    """
    Context manager that prints elapsed time.

    Usage:
        with timer("forward pass"):
            output = model(x)
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Elapsed: {elapsed * 1000:.2f} ms")


def measure_throughput(
    model:  nn.Module,
    device: torch.device,
    size:   int   = 256,
    batch:  int   = 8,
    warmup: int   = 10,
    runs:   int   = 50,
) -> Dict[str, float]:
    """
    Measure model throughput: FPS and latency.

    Returns:
        dict with 'fps', 'latency_ms', 'gpu_memory_mb'
    """
    model.eval()
    dummy = torch.randn(batch, 3, size, size, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    avg_time = np.mean(times)
    fps = batch / avg_time

    stats = {
        "fps":         fps,
        "latency_ms":  avg_time * 1000 / batch,
    }
    if device.type == "cuda":
        stats["gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1e6

    print(f"\n  Throughput: {fps:.1f} img/s  |  Latency: {stats['latency_ms']:.2f} ms/img")
    if "gpu_memory_mb" in stats:
        print(f"  GPU Memory: {stats['gpu_memory_mb']:.0f} MB")

    return stats


# ════════════════════════════════════════════════════════════════
#  Checkpoint Utilities
# ════════════════════════════════════════════════════════════════

def load_model_weights(
    model:            nn.Module,
    checkpoint_path:  str,
    strict:           bool = True,
    device:           Optional[torch.device] = None,
) -> int:
    """
    Load only model weights from a checkpoint (ignores optimizer etc.).

    Returns:
        epoch: the epoch at which the checkpoint was saved
    """
    dev   = device or torch.device("cpu")
    state = torch.load(checkpoint_path, map_location=dev)

    # Handle DataParallel prefix
    model_state = state.get("model", state)
    if list(model_state.keys())[0].startswith("module."):
        model_state = {k[7:]: v for k, v in model_state.items()}

    model.load_state_dict(model_state, strict=strict)
    epoch = state.get("epoch", 0)
    psnr  = state.get("best_psnr", 0.0)
    print(f"Loaded weights from epoch {epoch} (PSNR={psnr:.2f})")
    return epoch


def list_checkpoints(ckpt_dir: str) -> list:
    """List all .pth checkpoint files sorted by modification time."""
    d = Path(ckpt_dir)
    ckpts = sorted(d.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    return [str(p) for p in ckpts]


# ════════════════════════════════════════════════════════════════
#  Visualise helpers (shared between inference and evaluation)
# ════════════════════════════════════════════════════════════════

def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Convert a (C, H, W) or (1, C, H, W) tensor to (H, W, C) uint8.
    Reverses ImageNet normalisation for 3-channel images.
    Handles 1-channel maps by squeezing.
    """
    if t.ndim == 4:
        t = t.squeeze(0)
    t = t.detach().cpu().float()

    if t.shape[0] == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t    = t * std + mean
        img  = t.permute(1, 2, 0).clamp(0, 1).numpy()
        return (img * 255).astype(np.uint8)
    else:
        img = t.squeeze(0).clamp(0, 1).numpy()
        return (img * 255).astype(np.uint8)
