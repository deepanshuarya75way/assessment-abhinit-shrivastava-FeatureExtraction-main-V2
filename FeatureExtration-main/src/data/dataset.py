"""
src/data/dataset.py
===================
Fixed dataset loader (Albumentations v2 compatible)
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────
#  Paired Dataset
# ─────────────────────────────────────────────────────────────

class UnderwaterPairedDataset(Dataset):

    def __init__(
        self,
        degraded_dir: str,
        reference_dir: str,
        image_size: int = 256,
        augment: bool = True,
        split: str = "train",
    ):
        super().__init__()

        self.degraded_dir = Path(degraded_dir)
        self.reference_dir = Path(reference_dir)
        self.image_size = image_size
        self.augment = augment and (split == "train")

        self.pairs = self._build_pairs()
        assert len(self.pairs) > 0, "No image pairs found!"

        self.transform_shared, self.transform_img = self._build_transforms()

    def _build_pairs(self) -> List[Tuple[Path, Path]]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        degraded_files = sorted(
            p for p in self.degraded_dir.iterdir() if p.suffix.lower() in exts
        )

        pairs = []
        for deg in degraded_files:
            for ext in exts:
                ref = self.reference_dir / (deg.stem + ext)
                if ref.exists():
                    pairs.append((deg, ref))
                    break

        return pairs

    def _build_transforms(self):

        if self.augment:
            shared = A.Compose(
                [
                    A.RandomResizedCrop(
                        size=(self.image_size, self.image_size),  # ✅ FIX
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10, p=0.3),
                ],
                additional_targets={"reference": "image"},
            )
        else:
            shared = A.Compose(
                [
                    A.Resize(self.image_size, self.image_size),
                ],
                additional_targets={"reference": "image"},
            )

        to_tensor = A.Compose(
            [
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )

        return shared, to_tensor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        deg_path, ref_path = self.pairs[idx]

        deg_img = self._load_image(deg_path)
        ref_img = self._load_image(ref_path)

        result = self.transform_shared(image=deg_img, reference=ref_img)

        deg_img = result["image"]
        ref_img = result["reference"]

        deg_tensor = self.transform_img(image=deg_img)["image"]
        ref_tensor = self.transform_img(image=ref_img)["image"]

        return {
            "degraded": deg_tensor,
            "reference": ref_tensor,
        }

    @staticmethod
    def _load_image(path: Path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────
#  Unpaired Dataset
# ─────────────────────────────────────────────────────────────

class UnderwaterUnpairedDataset(Dataset):

    def __init__(self, image_dir: str, image_size: int = 256, augment=False):
        super().__init__()

        self.image_dir = Path(image_dir)
        self.image_size = image_size

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        self.images = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in exts
        )

        assert len(self.images) > 0, "No images found!"

        if augment:
            self.transform = A.Compose(
                [
                    A.RandomResizedCrop(
                        size=(image_size, image_size),  # ✅ FIX
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        path = self.images[idx]

        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = self.transform(image=img)["image"]

        return {"degraded": tensor}


# ─────────────────────────────────────────────────────────────
#  Dataset Factory
# ─────────────────────────────────────────────────────────────

class DatasetFactory:

    _DIR_MAP = {
        "uieb": ("raw-890", "reference-890"),
        "euvp_paired": ("trainA", "trainB"),
        "ufo120": ("lrd", "gt"),
    }

    def __init__(self, config: dict):
        self.config = config
        self.root = Path(config["data"]["root"])

    def build(self):

        name = self.config["data"]["dataset"]
        img_size = self.config["data"]["image_size"]

        if name in self._DIR_MAP:

            deg_sub, ref_sub = self._DIR_MAP[name]
            dataset_root = self.root / name

            dataset = UnderwaterPairedDataset(
                degraded_dir=str(dataset_root / deg_sub),
                reference_dir=str(dataset_root / ref_sub),
                image_size=img_size,
                augment=True,
            )

        else:
            raise ValueError("Unsupported dataset")

        train_ds, val_ds, test_ds = self._split(dataset)

        return (
            DataLoader(train_ds, batch_size=16, shuffle=True),
            DataLoader(val_ds, batch_size=16),
            DataLoader(test_ds, batch_size=16),
        )

    @staticmethod
    def _split(dataset):

        n = len(dataset)
        n_tr = int(n * 0.8)
        n_val = int(n * 0.1)
        n_te = n - n_tr - n_val

        return random_split(dataset, [n_tr, n_val, n_te])