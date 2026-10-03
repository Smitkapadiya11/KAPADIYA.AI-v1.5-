#!/usr/bin/env python3
"""KAPADIYA.AI v1.5 TMKB - single-image try-on CLI."""

import argparse
import sys
from pathlib import Path

from PIL import Image

from . import MODEL_NAME
from .pipeline import PRESETS, RESOLUTIONS, TryOnPipeline


def main():
    parser = argparse.ArgumentParser(
        description=MODEL_NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    kapadiya-tryon --weights-dir ./weights \
        --person-image examples/data/model.webp \
        --garment-image examples/data/garment.webp \
        --category tops --preset fast
        """,
    )
    parser.add_argument("--weights-dir", type=str, required=True, help="Directory containing model weights")
    parser.add_argument("--person-image", type=str, required=True, help="Path to person image")
    parser.add_argument("--garment-image", type=str, required=True, help="Path to garment image")
    parser.add_argument("--category", type=str, choices=["tops", "bottoms", "one-pieces"], required=True)
    parser.add_argument(
        "--garment-photo-type",
        type=str,
        choices=["model", "flat-lay"],
        default="model",
        help="'model' if worn by person, 'flat-lay' for product shots",
    )
    parser.add_argument("--preset", type=str, choices=list(PRESETS), default="balanced", help="Speed/quality preset")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of output images (1-4)")
    parser.add_argument("--num-timesteps", type=int, default=None, help="Override preset step count")
    parser.add_argument("--cfg-steps", type=int, default=-1, help="Apply CFG only on first N steps (override)")
    parser.add_argument("--resolution", type=str, choices=list(RESOLUTIONS), default=None, help="Override resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override CFG strength")
    parser.add_argument(
        "--no-segmentation-free",
        action="store_false",
        dest="segmentation_free",
        default=True,
        help="Disable segmentation-free mode",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--low-vram", action="store_true", default=None, help="Force low-VRAM mode (auto on <5 GB GPUs)")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile (Linux + Triton)")
    args = parser.parse_args()

    for p in (args.person_image, args.garment_image):
        if not Path(p).exists():
            sys.exit(f"Error: image not found: {p}")
    if not Path(args.weights_dir).exists():
        sys.exit(
            f"Error: weights directory not found: {args.weights_dir}\n"
            f"Run: python scripts/download_weights.py --weights-dir {args.weights_dir}"
        )

    person_image = Image.open(args.person_image).convert("RGB")
    garment_image = Image.open(args.garment_image).convert("RGB")

    print(f"Loading {MODEL_NAME} from {args.weights_dir}...")
    pipeline = TryOnPipeline(
        weights_dir=args.weights_dir,
        device=args.device,
        dtype=args.dtype,
        low_vram=args.low_vram,
        compile_model=args.compile,
    )

    result = pipeline(
        person_image=person_image,
        garment_image=garment_image,
        category=args.category,
        garment_photo_type=args.garment_photo_type,
        preset=args.preset,
        num_samples=args.num_samples,
        num_timesteps=args.num_timesteps,
        cfg_steps=args.cfg_steps,
        resolution=args.resolution,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        segmentation_free=args.segmentation_free,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, output_image in enumerate(result.images):
        output_path = output_dir / f"output_{i:02d}.png"
        output_image.save(output_path)
        print(f"Saved: {output_path}")

    print("Timings: " + " | ".join(f"{k} {v:.2f}s" for k, v in result.timings.items()))


if __name__ == "__main__":
    main()
