#!/usr/bin/env python3
"""
scripts/download_dataset.py
============================
Dataset download and preparation helpers.

Supported datasets:
  - UIEB     (950 paired images — recommended for supervised training)
  - EUVP     (12K images, paired + unpaired subsets)
  - UFO-120  (1500 paired, high-resolution)
  - RUIE     (4000+ real underwater, 3 subsets, no ground truth)

Usage:
    python scripts/download_dataset.py --dataset uieb   --output data/
    python scripts/download_dataset.py --dataset euvp   --output data/
    python scripts/download_dataset.py --dataset ufo120 --output data/
    python scripts/download_dataset.py --dataset ruie   --output data/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  Dataset registry
# ─────────────────────────────────────────────────────────────

DATASETS = {
    "uieb": {
        "name":        "UIEB (Underwater Image Enhancement Benchmark)",
        "url":         "https://li-chongyi.github.io/proj_benchmark.html",
        "gdrive_id":   "1QQiRPVYZBkRD4xb3LCzlrqRBVMtdXAkW",
        "description": (
            "950 paired images (raw underwater + reference). "
            "Gold standard for supervised UIE. "
            "~90MB compressed."
        ),
        "structure": {
            "degraded": "raw-890/",
            "reference": "reference-890/",
        },
        "note": (
            "Manual download required from: "
            "https://li-chongyi.github.io/proj_benchmark.html"
        ),
    },
    "euvp": {
        "name":        "EUVP (Enhancing Underwater Visual Perception)",
        "url":         "https://irvlab.cs.umn.edu/resources/euvp-dataset",
        "description": (
            "Paired (~3K) + unpaired (~9K) images. "
            "Generated with Fast-CycleGAN. Good for semi-supervised training."
        ),
        "structure": {
            "paired_degraded": "paired/trainA/",
            "paired_reference": "paired/trainB/",
            "unpaired": "unpaired/trainA/",
        },
        "note": "Download from: https://irvlab.cs.umn.edu/resources/euvp-dataset",
    },
    "ufo120": {
        "name":        "UFO-120",
        "url":         "https://irvlab.cs.umn.edu/resources/ufo-120",
        "description": (
            "1500 paired images at high resolution (640×480). "
            "Includes depth maps. Suitable for multi-task learning."
        ),
        "structure": {
            "degraded": "lrd/",
            "reference": "gt/",
        },
        "note": "Download from: https://irvlab.cs.umn.edu/resources/ufo-120",
    },
    "ruie": {
        "name":        "RUIE (Real Underwater Image Enhancement)",
        "url":         "https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark",
        "description": (
            "4000+ real underwater images. 3 subsets: "
            "UCCS (colour cast), UIQS (quality), UHTS (turbidity). "
            "NO ground truth — use UIQM/UCIQE for evaluation."
        ),
        "structure": {
            "uccs": "UCCS/",
            "uiqs": "UIQS/",
            "uhts": "UHTS/",
        },
        "note": "Download from GitHub: dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark",
    },
    "squid": {
        "name":        "SQUID (Stereo Quantitative Underwater Image Dataset)",
        "url":         "http://csms.haifa.ac.il/profiles/tTreibitz/datasets/ambient_forwardlooking/index.html",
        "description": (
            "57 stereo scenes with depth information. "
            "Useful for physics-based / depth-aware methods."
        ),
        "note": "Manual download required from dataset page.",
    },
}


def print_dataset_info(name: str):
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        return
    d = DATASETS[name]
    print(f"\n{'='*60}")
    print(f"  {d['name']}")
    print(f"{'='*60}")
    print(f"  URL         : {d['url']}")
    print(f"  Description : {d['description']}")
    if "structure" in d:
        print(f"  Structure:")
        for k, v in d["structure"].items():
            print(f"              {k:20s} → {v}")
    print(f"  Note        : {d.get('note', 'N/A')}")
    print()


def list_all_datasets():
    print("\n  Available Datasets\n  " + "─"*45)
    for name, info in DATASETS.items():
        print(f"\n  [{name}]")
        print(f"    {info['description'][:80]}...")
        print(f"    {info.get('note', '')}")
    print()


def prepare_directory_structure(output_dir: str, dataset: str):
    """
    Create the expected directory structure for a dataset so that
    the DatasetFactory can locate images correctly.
    """
    base = Path(output_dir) / dataset

    if dataset not in DATASETS:
        return
    structure = DATASETS[dataset].get("structure", {})

    for key, subpath in structure.items():
        full_path = base / subpath
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {full_path}")

    print(f"\n  Place your {DATASETS[dataset]['name']} images in the directories above.")


def verify_dataset(dataset_dir: str, dataset: str) -> bool:
    """
    Verify that a downloaded dataset has the expected structure
    and contains images.
    """
    if dataset not in DATASETS:
        return False

    base      = Path(dataset_dir) / dataset
    structure = DATASETS[dataset].get("structure", {})
    exts      = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    all_ok    = True

    print(f"\n  Verifying {dataset}:")
    for key, subpath in structure.items():
        path   = base / subpath
        exists = path.exists()
        n_imgs = len([f for f in path.iterdir() if f.suffix.lower() in exts]) if exists else 0
        status = "✓" if exists and n_imgs > 0 else "✗"
        print(f"    {status} {key:<25} {n_imgs:>5} images  [{path}]")
        if not exists or n_imgs == 0:
            all_ok = False

    return all_ok


def parse_args():
    p = argparse.ArgumentParser(description="Dataset download and preparation")
    p.add_argument("--dataset",  type=str, choices=list(DATASETS.keys()) + ["all"],
                   help="Dataset to prepare")
    p.add_argument("--output",   type=str, default="data/",
                   help="Output directory")
    p.add_argument("--info",     action="store_true",
                   help="Print information about a dataset")
    p.add_argument("--list",     action="store_true",
                   help="List all available datasets")
    p.add_argument("--verify",   action="store_true",
                   help="Verify an existing dataset download")
    p.add_argument("--prepare",  action="store_true",
                   help="Create directory structure for manual download")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_all_datasets()
        return

    if args.dataset is None:
        print("Please specify --dataset or use --list to see options.")
        return

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        if args.info or (not args.verify and not args.prepare):
            print_dataset_info(ds)

        if args.prepare:
            print(f"\nPreparing directory structure for: {ds}")
            prepare_directory_structure(args.output, ds)

        if args.verify:
            ok = verify_dataset(args.output, ds)
            sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
