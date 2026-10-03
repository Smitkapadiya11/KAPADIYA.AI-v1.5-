#!/usr/bin/env python3
"""
Benchmark KAPADIYA.AI v1.5 TMKB presets on this machine.

Usage:
    python scripts/benchmark.py --weights-dir ./weights [--presets live fast balanced] [--runs 3]

Reports per-stage latency (warm runs, garment cache hit) and peak VRAM, and saves one
output per preset so speed and quality can be compared side by side.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from kapadiya_ai import MODEL_NAME, PRESETS, TryOnPipeline


def main():
    parser = argparse.ArgumentParser(description=f"Benchmark {MODEL_NAME}")
    parser.add_argument("--weights-dir", type=str, required=True)
    parser.add_argument("--person-image", type=str, default="examples/data/model.webp")
    parser.add_argument("--garment-image", type=str, default="examples/data/garment.webp")
    parser.add_argument("--category", type=str, default="tops")
    parser.add_argument("--presets", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--low-vram", action="store_true", default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark")
    args = parser.parse_args()

    person = Image.open(args.person_image).convert("RGB")
    garment = Image.open(args.garment_image).convert("RGB")
    pipe = TryOnPipeline(args.weights_dir, device=args.device, dtype=args.dtype, low_vram=args.low_vram)
    cuda = pipe.device.type == "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for preset in args.presets:
        pipe.warmup(PRESETS[preset]["resolution"])
        pipe(person, garment, category=args.category, preset=preset, use_tqdm=False)  # fills garment cache
        if cuda:
            torch.cuda.reset_peak_memory_stats(pipe.device)

        timings = []
        for _ in range(args.runs):
            out = pipe(person, garment, category=args.category, preset=preset, use_tqdm=False)
            timings.append(out.timings)
        out.images[0].save(out_dir / f"{preset}.png")

        avg = {k: sum(t[k] for t in timings) / len(timings) for k in timings[0]}
        peak = torch.cuda.max_memory_allocated(pipe.device) / 1024**3 if cuda else float("nan")
        rows.append((preset, avg, peak))

    print(f"\n{MODEL_NAME} on {torch.cuda.get_device_name(pipe.device) if cuda else 'CPU'} ({pipe.inference_dtype})")
    print(f"{'preset':<10}{'person prep':>13}{'sampling':>11}{'total':>9}{'peak VRAM':>12}")
    for preset, avg, peak in rows:
        print(
            f"{preset:<10}{avg['person_preprocess']:>12.2f}s{avg['sampling']:>10.2f}s"
            f"{avg['total']:>8.2f}s{peak:>10.2f}GB"
        )
    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
