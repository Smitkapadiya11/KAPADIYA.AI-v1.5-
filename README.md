# KAPADIYA.AI v1.5 TMKB

**Fast, low-memory virtual try-on that runs on an ordinary laptop: 8 GB RAM and a 4 GB GPU.**

Give it a photo of a person and a photo of a garment, and it produces a photorealistic image of that person wearing the garment. It also includes a **live webcam mode** that keeps re-dressing you while you move.

KAPADIYA.AI v1.5 TMKB is an inference-optimized edition of [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5) (Apache-2.0). It is a ~1B-parameter MMDiT that generates directly in pixel space and needs no segmentation masks. It uses the original FASHN weights, which have **not** been retrained. All of the speed and memory gains below come from changes to how the model is loaded and sampled. See [NOTICE](NOTICE) for full attribution.

---

## What's new vs. FASHN VTON v1.5

| Area | Upstream | KAPADIYA.AI v1.5 TMKB |
|------|----------|------------------------|
| Model loading | Random-init fp32 model, then copy in the weights (~6 GB host RAM peak) | Meta-device model filled tensor by tensor from safetensors (~1 tensor of host RAM) |
| Precision | bf16 or **fp32** (fp32 doesn't fit in 4 GB) | bf16 on RTX 30/40, **fp16** on GTX 16xx / RTX 20xx, fp32 on CPU |
| Low-VRAM | – | Auto on GPUs under 5 GB: pose and parser models run on CPU, and CFG runs as two batch-1 passes |
| Guidance | CFG on every step (2 model passes per step) | **Interval CFG**: guidance only on the first N steps, 1 pass per step after that |
| Resolution | 864×576 | 864×576, or **576×384 preview** (2.25× fewer tokens) |
| Repeated garment | Re-runs pose and parsing every call | **Cached**, so only the person is processed |
| Positional embeddings | Recomputed in float64 every step | Computed once per resolution |
| Video / live | – | **Warm start**: each frame starts from the previous result, so it needs fewer steps and flickers less |
| Tooling | Example script | `kapadiya-tryon` CLI, `kapadiya-live` webcam app, `scripts/benchmark.py` |

### Presets

| Preset | Steps | CFG steps | Resolution | Model passes* | Use for |
|--------|------:|----------:|-----------:|--------------:|---------|
| `quality`  | 50 | all | 864×576 | 99 | final renders |
| `balanced` | 30 | all | 864×576 | 59 | **identical to upstream defaults** |
| `fast`     | 16 | 8   | 864×576 | 24 (~2.5× fewer) | interactive apps |
| `live`     | 8  | 3   | 576×384 | 11 at ~⅖ the cost each | webcam preview |
| `live` + warm start (strength 0.6) | 5 of 8 | 0 | 576×384 | 5 at ~⅖ the cost each | continuous frames |

\* Counts per-sample model forward passes. These are compute counts, not measured timings. Measure your own hardware with `scripts/benchmark.py`.

> **Honest limits.** This is not a real-time video model. Diffusion try-on with a 1B-parameter network takes seconds per frame on a laptop GPU. The live mode is *near-live*: the camera feed runs at full FPS and the try-on panel refreshes as each frame finishes. The `preview` resolution and very low step counts trade away some quality, so use `fast` or `balanced` for final images.

---

## Installation

```bash
git clone https://github.com/Smitkapadiya11/KAPADIYA.AI-v1.5-.git
cd KAPADIYA.AI-v1.5-
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# Install CUDA PyTorch first (pick the command for your CUDA from https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

The install includes `onnxruntime-gpu`. If you don't have a GPU, run `pip uninstall onnxruntime-gpu && pip install onnxruntime` (pose detection falls back to CPU automatically).

### Model weights (~2 GB)

```bash
python scripts/download_weights.py --weights-dir ./weights
```

This downloads `model.safetensors` from [fashn-ai/fashn-vton-1.5](https://huggingface.co/fashn-ai/fashn-vton-1.5) and the DWPose ONNX models. The human-parser weights (~244 MB) go to the Hugging Face cache.

---

## Usage

### Python

```python
from kapadiya_ai import TryOnPipeline
from PIL import Image

pipe = TryOnPipeline(weights_dir="./weights")  # auto device / dtype / low-VRAM

person = Image.open("examples/data/model.webp").convert("RGB")
garment = Image.open("examples/data/garment.webp").convert("RGB")

result = pipe(person, garment, category="tops", preset="fast")
result.images[0].save("output.png")
print(result.timings)  # per-stage seconds
```

### CLI

```bash
kapadiya-tryon --weights-dir ./weights \
    --person-image examples/data/model.webp \
    --garment-image examples/data/garment.webp \
    --category tops --preset fast
```

Useful flags: `--preset live|fast|balanced|quality`, `--resolution preview|full`, `--num-timesteps N`, `--cfg-steps N`, `--dtype float16`, `--low-vram`, `--garment-photo-type flat-lay`.

### Live webcam try-on

```bash
kapadiya-live --weights-dir ./weights --garment-image my_shirt.jpg --category tops
```

Keys: `q` quit · `s` save frame · `w` toggle warm start · `c` cycle category.
Lower `--strength` (e.g. 0.4) is faster and more stable but slower to follow your movement. Raise it toward 1.0 for more accurate frames.

### Benchmark your machine

```bash
python scripts/benchmark.py --weights-dir ./weights
```

This prints per-preset latency and peak VRAM, and saves one image per preset to `outputs/benchmark/` so you can compare quality.

---

## Hardware notes

- **4 GB VRAM:** the try-on model takes ~1.9 GB in bf16/fp16. Low-VRAM mode turns on automatically and keeps the pose and parser models on the CPU.
- **8 GB RAM:** weights stream straight to the GPU, one tensor at a time, so no full fp32 copy is ever built in RAM.
- **GTX 16xx / RTX 20xx** (no bf16): runs in fp16. If outputs come out black or noisy, use `--dtype float32` on CPU or a bf16-capable GPU.
- **Windows:** `--compile` needs Triton, which is usually unavailable on Windows. The flag is ignored with a warning.

## Categories

| Category | Description |
|----------|-------------|
| `tops` | Upper body: t-shirts, blouses, jackets |
| `bottoms` | Lower body: pants, skirts, shorts |
| `one-pieces` | Full body: dresses, jumpsuits |

## Tests

```bash
pip install -e ".[dev]" && pytest tests
```

The optimization tests use a tiny random model, so they run on CPU in seconds. They check that streaming load matches the original weights, that output is resolution-agnostic, that cached RoPE gives the same result as recomputing it, that low-VRAM CFG matches batched CFG, and that interval CFG and warm start use the expected number of passes.

---

## License & credits

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Model architecture and weights: **FASHN VTON v1.5** by [FASHN AI](https://fashn.ai), which you should cite if you use this in research:

```bibtex
@article{bochman2026fashnvton,
  title={FASHN VTON v1.5: Efficient Maskless Virtual Try-On in Pixel Space},
  author={Bochman, Dan and Bochman, Aya},
  journal={arXiv preprint},
  year={2026}
}
```

Third-party components: [FLUX.1](https://github.com/black-forest-labs/flux) (Apache-2.0), [DWPose](https://github.com/IDEA-Research/DWPose) (Apache-2.0), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0), and [fashn-human-parser](https://github.com/fashn-AI/fashn-human-parser) ([license](https://github.com/fashn-AI/fashn-human-parser?tab=readme-ov-file#license)).
