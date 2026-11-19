"""
src/training/trainer.py
========================
Training Pipeline

Features:
  - Mixed Precision (AMP) training
  - Gradient clipping
  - Warmup + cosine LR scheduling
  - Tensorboard & optional W&B logging
  - Checkpoint saving (best + latest)
  - Validation loop with early stopping
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.feature_extractor import UnderwaterFeatureExtractor
from ..losses.losses import UnderwaterCombinedLoss
from ..evaluation.metrics import compute_psnr, compute_ssim


# ════════════════════════════════════════════════════════════════
#  Learning Rate Schedulers
# ════════════════════════════════════════════════════════════════

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to `min_lr`.
    """

    def __init__(
        self,
        optimizer:     optim.Optimizer,
        warmup_epochs: int,
        total_epochs:  int,
        min_lr:        float = 1e-6,
        last_epoch:    int   = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        import math
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]


def build_optimizer(config: dict, model: nn.Module) -> optim.Optimizer:
    opt_cfg = config["training"]["optimizer"]
    opt_type = opt_cfg["type"].lower()
    lr = opt_cfg["lr"]
    wd = opt_cfg["weight_decay"]

    if opt_type == "adamw":
        return optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
    elif opt_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def build_scheduler(
    config: dict, optimizer: optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    sched_cfg    = config["training"]["scheduler"]
    sched_type   = sched_cfg["type"].lower()
    total_epochs = config["training"]["epochs"]

    if sched_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=sched_cfg.get("min_lr", 1e-6),
        )
    elif sched_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=sched_cfg.get("warmup_epochs", 5),
            total_epochs=total_epochs,
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
    elif sched_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 30),
            gamma=sched_cfg.get("gamma", 0.5),
        )
    elif sched_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5,
            factor=sched_cfg.get("gamma", 0.5),
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")


# ════════════════════════════════════════════════════════════════
#  Trainer
# ════════════════════════════════════════════════════════════════

class Trainer:
    """
    Manages the complete training loop.

    Usage:
        trainer = Trainer(config, model, train_loader, val_loader)
        trainer.train()
    """

    def __init__(
        self,
        config:       dict,
        model:        UnderwaterFeatureExtractor,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       Optional[torch.device] = None,
    ):
        self.config       = config
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device or torch.device(
            config.get("project", {}).get("device", "cuda")
            if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        # Training params
        t = config["training"]
        self.epochs        = t["epochs"]
        self.grad_clip     = t["gradient_clip"]
        self.log_interval  = t["log_interval"]
        self.save_interval = t["save_interval"]
        self.val_interval  = t["val_interval"]
        self.use_amp       = config.get("project", {}).get("amp", True)

        # Paths
        self.ckpt_dir  = Path(config["paths"]["checkpoints"])
        self.log_dir   = Path(config["paths"]["logs"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt = config["paths"]["best_model"]

        # Optimiser / scheduler / loss
        self.optimizer  = build_optimizer(config, model)
        self.scheduler  = build_scheduler(config, self.optimizer)
        self.criterion  = UnderwaterCombinedLoss(config).to(self.device)
        self.scaler     = GradScaler(enabled=self.use_amp)

        # Logging
        self.writer = None
        if config.get("logging", {}).get("use_tensorboard", True):
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.global_step  = 0
        self.best_psnr    = 0.0
        self.best_epoch   = 0

    # ──────────────────────────────────────────────────────────
    #  Main train loop
    # ──────────────────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}")
        print(f"  Training: {self.config['project']['name']}")
        print(f"  Epochs: {self.epochs}  |  Device: {self.device}")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            train_losses = self._train_epoch(epoch)
            self._log_epoch(train_losses, epoch, prefix="train")

            if epoch % self.val_interval == 0:
                val_metrics = self._val_epoch(epoch)
                self._log_epoch(val_metrics, epoch, prefix="val")

                # Save best model by PSNR
                psnr = val_metrics.get("metric/psnr", 0.0)
                if psnr > self.best_psnr:
                    self.best_psnr  = psnr
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, tag="best")
                    print(f"  ✓ New best PSNR: {psnr:.2f} dB (epoch {epoch})")

            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch, tag=f"epoch_{epoch:03d}")

            # Scheduler step
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(train_losses["loss/total"])
            else:
                self.scheduler.step()

        print(f"\nTraining complete.")
        print(f"Best PSNR: {self.best_psnr:.2f} dB at epoch {self.best_epoch}")
        if self.writer:
            self.writer.close()

    # ──────────────────────────────────────────────────────────
    #  Training epoch
    # ──────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_losses: Dict[str, float] = {}
        n_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            degraded  = batch["degraded"].to(self.device)   # (B, 3, H, W)
            reference = batch["reference"].to(self.device)  # (B, 3, H, W)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                model_out = self.model(degraded)
                # Pass input through for losses that need it
                model_out["input"] = degraded
                loss, loss_dict = self.criterion(model_out, reference)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v

            # Per-step logging
            if self.global_step % self.log_interval == 0 and self.writer:
                for k, v in loss_dict.items():
                    self.writer.add_scalar(k + "/step", v, self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", lr, self.global_step)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            self.global_step += 1

        # Average losses over epoch
        return {k: v / n_batches for k, v in total_losses.items()}

    # ──────────────────────────────────────────────────────────
    #  Validation epoch
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        psnr_sum = ssim_sum = 0.0
        n = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)
        for batch in pbar:
            degraded  = batch["degraded"].to(self.device)
            reference = batch["reference"].to(self.device)

            model_out = self.model(degraded)
            enhanced  = model_out["enhanced"]

            # PSNR / SSIM in [0, 1]
            psnr_sum += compute_psnr(enhanced, reference)
            ssim_sum += compute_ssim(enhanced, reference)
            n += 1

        return {
            "metric/psnr": psnr_sum / max(1, n),
            "metric/ssim": ssim_sum / max(1, n),
        }

    # ──────────────────────────────────────────────────────────
    #  Logging
    # ──────────────────────────────────────────────────────────

    def _log_epoch(self, metrics: Dict[str, float], epoch: int, prefix: str):
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, epoch)
        # Console
        summary = "  ".join(f"{k.split('/')[-1]}: {v:.4f}" for k, v in metrics.items())
        print(f"[E{epoch:03d}/{self.epochs}] {prefix.upper():5s} | {summary}")

    # ──────────────────────────────────────────────────────────
    #  Checkpointing
    # ──────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, tag: str = "latest"):
        state = {
            "epoch":       epoch,
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "best_psnr":   self.best_psnr,
            "global_step": self.global_step,
            "config":      self.config,
        }
        if tag == "best":
            path = self.best_ckpt
        else:
            path = self.ckpt_dir / f"checkpoint_{tag}.pth"

        torch.save(state, path)

    def load_checkpoint(self, path: str, resume: bool = True):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        if resume:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.best_psnr   = state.get("best_psnr", 0.0)
            self.global_step = state.get("global_step", 0)
        return state.get("epoch", 0)
