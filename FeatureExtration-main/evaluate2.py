#!/usr/bin/env python3
"""
evaluate.py
============
Full evaluation on a dataset split.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth --split test
    python evaluate.py --checkpoint checkpoints/best_model.pth --save_visuals
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.utils              import load_config, set_seed, get_device
from src.data.dataset              import DatasetFactory
from src.models.feature_extractor  import build_model, UnderwaterFeatureExtractor
from src.evaluation.metrics        import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Underwater Feature Extractor")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pth file")
    parser.add_argument("--config",     type=str, default="config/config.yaml")
    parser.add_argument("--split",      type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--save_visuals", action="store_true",
                        help="Save visualisation panels")
    parser.add_argument("--visual_dir",   type=str, default="outputs/eval_visuals")
    parser.add_argument("--visual_count", type=int, default=20)
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    set_seed(config.get("project", {}).get("seed", 42))
    device = get_device(config.get("project", {}).get("device", "cuda"))

    # ── Load model ────────────────────────────────────────────
    state = torch.load(args.checkpoint, map_location=device)
    cfg   = state.get("config", config)
    model = build_model(cfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    epoch     = state.get("epoch", "?")
    best_psnr = state.get("best_psnr", 0.0)
    print(f"Loaded checkpoint  epoch={epoch}  best_psnr={best_psnr:.2f}")

    # ── Data ──────────────────────────────────────────────────
    factory = DatasetFactory(config)
    try:
        _, val_loader, test_loader = factory.build()
        loader = test_loader if args.split == "test" else val_loader
    except Exception as e:
        print(f"[WARNING] Dataset load failed: {e}")
        print("Using dummy data for demonstration.")
        loader = _dummy_loader(config)

    # ── Evaluate ──────────────────────────────────────────────
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(
        loader,
        save_dir   = args.visual_dir if args.save_visuals else None,
        save_count = args.visual_count,
    )

    # ── Print results ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  Evaluation Results  [{args.split.upper()} split]")
    print("=" * 50)
    for name, val in metrics.items():
        unit = " dB" if name == "psnr" else ""
        print(f"  {name.upper():<10}: {val:.4f}{unit}")
    print("=" * 50)

    if args.save_visuals:
        print(f"\n  Visuals saved to: {args.visual_dir}")


def _dummy_loader(config):
    import torch
    from torch.utils.data import DataLoader

    class DummyDS(torch.utils.data.Dataset):
        def __init__(self, n, sz):
            self.n, self.sz = n, sz
        def __len__(self):
            return self.n
        def __getitem__(self, i):
            return {
                "degraded":   torch.rand(3, self.sz, self.sz),
                "reference":  torch.rand(3, self.sz, self.sz),
            }

    sz = config["data"]["image_size"]
    return DataLoader(DummyDS(20, sz), batch_size=1, shuffle=False)


if __name__ == "__main__":
    main()
