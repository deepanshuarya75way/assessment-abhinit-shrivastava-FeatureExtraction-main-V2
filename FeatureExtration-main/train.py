#!/usr/bin/env python3
"""
train.py
=========
Main training entry point.

Usage:
    python train.py
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --resume checkpoints/checkpoint_epoch_050.pth
    python train.py --epochs 200 --lr 0.0002 --batch_size 16
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Make src importable
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.utils      import load_config, set_seed, get_device, model_summary
from src.data.dataset      import DatasetFactory
from src.models.feature_extractor import build_model
from src.training.trainer  import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Underwater Feature Extractor"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    # Override individual config values from CLI
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--dataset",    type=str,   default=None,
                        help="Dataset name: uieb | euvp_paired | ufo120 | ruie")
    parser.add_argument("--data_root",  type=str,   default=None)
    parser.add_argument("--backbone",   type=str,   default=None,
                        help="resnet18 | resnet34 | resnet50")
    parser.add_argument("--no_amp",     action="store_true",
                        help="Disable Automatic Mixed Precision")
    parser.add_argument("--seed",       type=int,   default=None)
    return parser.parse_args()


def apply_cli_overrides(config: dict, args) -> dict:
    """Merge CLI arguments into config (CLI takes priority)."""
    if args.epochs     is not None: config["training"]["epochs"]         = args.epochs
    if args.lr         is not None: config["training"]["optimizer"]["lr"] = args.lr
    if args.batch_size is not None: config["training"]["batch_size"]      = args.batch_size
    if args.dataset    is not None: config["data"]["dataset"]             = args.dataset
    if args.data_root  is not None: config["data"]["root"]                = args.data_root
    if args.backbone   is not None: config["model"]["backbone"]           = args.backbone
    if args.no_amp:                 config["project"]["amp"]              = False
    if args.seed       is not None: config["project"]["seed"]             = args.seed
    return config


def main():
    args   = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # ── Reproducibility ────────────────────────────────────────
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # ── Device ─────────────────────────────────────────────────
    device = get_device(config.get("project", {}).get("device", "cuda"))

    # ── Data ───────────────────────────────────────────────────
    print(f"\nLoading dataset: {config['data']['dataset']} ...")
    factory = DatasetFactory(config)
    try:
        train_loader, val_loader, test_loader = factory.build()
        print(f"  Train: {len(train_loader.dataset):,} images")
        print(f"  Val:   {len(val_loader.dataset):,} images")
        print(f"  Test:  {len(test_loader.dataset):,} images")
    except Exception as e:
        print(f"\n[WARNING] Could not build dataset: {e}")
        print("  → Running with synthetic dummy data for development.\n")
        train_loader, val_loader, test_loader = _make_dummy_loaders(config)

    # ── Model ──────────────────────────────────────────────────
    print(f"\nBuilding model: {config['model']['backbone']} ...")
    model = build_model(config)
    model_summary(model)

    # ── Trainer ────────────────────────────────────────────────
    trainer = Trainer(
        config       = config,
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
    )

    # Resume from checkpoint
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume, resume=True)
        print(f"Resumed from epoch {start_epoch}")

    # ── Train ──────────────────────────────────────────────────
    trainer.train()

    # ── Final test evaluation ──────────────────────────────────
    print("\nRunning test set evaluation ...")
    from src.evaluation.metrics import Evaluator
    evaluator = Evaluator(model, device)
    test_metrics = evaluator.evaluate(
        test_loader,
        save_dir   = config["paths"]["outputs"] + "/test_visuals",
        save_count = config["evaluation"].get("visual_count", 20),
    )
    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k.upper():<8}: {v:.4f}")


# ─────────────────────────────────────────────────────────────
#  Dummy data loader (development / CI without real datasets)
# ─────────────────────────────────────────────────────────────

def _make_dummy_loaders(config: dict):
    """
    Creates dummy DataLoaders with random tensors so the training
    pipeline can be tested without downloading real datasets.
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    B  = 16
    sz = config["data"]["image_size"]
    bs = config["training"]["batch_size"]

    def _make(n):
        deg  = torch.rand(n, 3, sz, sz)
        ref  = torch.rand(n, 3, sz, sz)
        ds   = TensorDataset(deg, ref)
        # Wrap with dict-style batch using a custom collator
        return DataLoader(
            _DictDataset(deg, ref), batch_size=bs, shuffle=True, drop_last=True
        )

    return _make(64), _make(16), _make(16)


class _DictDataset(torch.utils.data.Dataset):
    def __init__(self, deg, ref):
        self.deg, self.ref = deg, ref
    def __len__(self):
        return len(self.deg)
    def __getitem__(self, i):
        return {"degraded": self.deg[i], "reference": self.ref[i]}


if __name__ == "__main__":
    main()
