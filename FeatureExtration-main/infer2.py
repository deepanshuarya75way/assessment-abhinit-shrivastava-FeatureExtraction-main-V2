#!/usr/bin/env python3
"""
infer.py
=========
Run inference on a single image or folder of images.

Usage:
    # Single image
    python infer.py --input image.jpg --checkpoint checkpoints/best_model.pth

    # Folder of images
    python infer.py --input data/test/ --checkpoint checkpoints/best_model.pth --folder

    # Save full visualisation panel
    python infer.py --input image.jpg --checkpoint checkpoints/best_model.pth --visualize

    # TTA (Test-Time Augmentation)
    python infer.py --input image.jpg --checkpoint checkpoints/best_model.pth --tta

    # Tiled inference for very large images
    python infer.py --input large_image.jpg --checkpoint checkpoints/best_model.pth --tiled
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Underwater Image Inference")
    p.add_argument("--input",      type=str, required=True,
                   help="Input image path or folder")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Model checkpoint (.pth)")
    p.add_argument("--output_dir", type=str, default="outputs/",
                   help="Output directory")
    p.add_argument("--image_size", type=int, default=512,
                   help="Inference image size")
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--visualize",  action="store_true",
                   help="Save full visualisation panel")
    p.add_argument("--compare",    action="store_true",
                   help="Save side-by-side before/after comparison")
    p.add_argument("--folder",     action="store_true",
                   help="Process entire folder")
    p.add_argument("--tta",        action="store_true",
                   help="Use test-time augmentation")
    p.add_argument("--tiled",      action="store_true",
                   help="Use tiled inference (for large images)")
    p.add_argument("--tile_size",  type=int, default=512)
    p.add_argument("--tile_overlap", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    from src.inference.inferencer import UnderwaterInferencer

    # ── Load model ────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    inferencer = UnderwaterInferencer.from_checkpoint(
        checkpoint_path = args.checkpoint,
        device          = args.device,
        image_size      = args.image_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference mode ────────────────────────────────────────
    if args.folder:
        # ── Folder mode ───────────────────────────────────────
        print(f"\nProcessing folder: {args.input}")
        inferencer.infer_folder(
            input_dir  = args.input,
            output_dir = str(output_dir),
            visualise  = args.visualize,
        )

    elif args.tiled:
        # ── Tiled mode (large images) ─────────────────────────
        print(f"\nTiled inference: {args.input}")
        enhanced = inferencer.infer_tiled(
            image_path = args.input,
            tile_size  = args.tile_size,
            overlap    = args.tile_overlap,
        )
        stem = Path(args.input).stem
        save_path = str(output_dir / f"{stem}_enhanced_tiled.png")
        from src.inference.inferencer import save_image
        save_image(enhanced, save_path)
        print(f"Saved: {save_path}")

    else:
        # ── Single image mode ─────────────────────────────────
        print(f"\nProcessing: {args.input}")
        result = inferencer.infer_single(args.input, use_tta=args.tta)

        stem = Path(args.input).stem

        # Always save enhanced image
        from src.inference.inferencer import save_image
        enh_path = str(output_dir / f"{stem}_enhanced.png")
        save_image(result["enhanced"], enh_path)
        print(f"Enhanced image saved: {enh_path}")

        # Visualisation panel
        if args.visualize:
            vis_path = str(output_dir / f"{stem}_visualisation.png")
            inferencer.save_visualisation(result, vis_path)

        # Side-by-side comparison
        if args.compare:
            cmp_path = str(output_dir / f"{stem}_comparison.png")
            inferencer.save_comparison(result, cmp_path)
            print(f"Comparison saved: {cmp_path}")

        # Print estimated quality metrics (no-reference)
        try:
            from src.evaluation.metrics import compute_uiqm, compute_uciqe
            uiqm  = compute_uiqm(result["enhanced"])
            uciqe = compute_uciqe(result["enhanced"])
            print(f"\nNo-reference quality metrics:")
            print(f"  UIQM : {uiqm:.4f}  (higher is better, typical range [-2, 3])")
            print(f"  UCIQE: {uciqe:.4f}  (higher is better, range [0, 1])")
        except Exception:
            pass   # metrics are optional


if __name__ == "__main__":
    main()
