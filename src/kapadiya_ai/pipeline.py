"""KAPADIYA.AI v1.5 TMKB inference pipeline.

Built on the FASHN VTON v1.5 pipeline (Apache-2.0). Changes vs. upstream:
- Streaming, meta-device model loading (low host RAM, no fp32 copy).
- Automatic precision: bf16 on Ampere+, fp16 on older CUDA GPUs, fp32 on CPU.
- Low-VRAM mode: aux models (pose, parser) on CPU, CFG run as two batch-1 passes.
- CFG interval: guidance only on the first N steps, conditional-only passes after.
- Resolution presets: "full" (864x576, native) and "preview" (576x384, ~3x fewer tokens).
- Garment preprocessing cache, so repeated try-ons of one garment only process the person.
- Warm start (init_image + strength) for temporally stable, cheaper live/video frames.
- Per-stage timings on every call.
"""

import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE, FashnHumanParser
from PIL import Image
from tqdm.auto import tqdm

from .dwpose import DWposeDetector, draw_pose
from .preprocessing import (
    BODY_COVERAGE_TO_FASHN_LABELS,
    FASHN_LABELS_TO_IDS,
    AspectPreserveResize,
    ResizePad,
    create_clothing_agnostic_image,
    create_garment_image,
)
from .tryon_mmdit import TryOnModel
from .utils import (
    get_dummy_dw_keypoints,
    get_rf_schedule,
    load_safetensors_streaming,
    normalize_uint8_to_neg1_1,
    numpy_to_torch,
    setup_logger,
    tensor_to_pil,
)

# (height, width); both must be multiples of the model patch size (12).
RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "full": (864, 576),  # native training resolution - best quality
    "preview": (576, 384),  # 2.25x fewer tokens per image; experimental, for live preview
}

# Sampling presets. "balanced" reproduces the upstream FASHN VTON v1.5 defaults exactly.
PRESETS: Dict[str, dict] = {
    "live": dict(num_timesteps=8, guidance_scale=1.5, cfg_steps=3, skip_cfg_last_n_steps=1, resolution="preview"),
    "fast": dict(num_timesteps=16, guidance_scale=1.5, cfg_steps=8, skip_cfg_last_n_steps=1, resolution="full"),
    "balanced": dict(num_timesteps=30, guidance_scale=1.5, cfg_steps=None, skip_cfg_last_n_steps=1, resolution="full"),
    "quality": dict(num_timesteps=50, guidance_scale=1.5, cfg_steps=None, skip_cfg_last_n_steps=1, resolution="full"),
}

LOW_VRAM_THRESHOLD_GB = 5.0
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class PipelineOutput:
    """Pipeline output container."""

    images: List[Image.Image]
    # Raw model-resolution output in [-1, 1] (padded); pass back as `init_image` for warm-starting the next frame.
    tensors: Optional[torch.Tensor] = None
    timings: Dict[str, float] = field(default_factory=dict)


class TryOnPipeline:
    """
    KAPADIYA.AI v1.5 TMKB try-on pipeline.

    Args:
        weights_dir: Directory containing model weights (model.safetensors, dwpose/)
        device: 'cuda', 'cpu', or None for auto-detect
        dtype: 'bfloat16' | 'float16' | 'float32' | None (auto)
        low_vram: Keep only the try-on model on the GPU and run CFG as two batch-1 passes.
            None = auto (enabled on GPUs with < 5 GB VRAM).
        compile_model: Try torch.compile on the try-on model (needs Triton; ignored if unavailable).
        logger: Optional logger instance

    Example:
        pipeline = TryOnPipeline(weights_dir="./weights")
        result = pipeline(person_image, garment_image, category="tops", preset="fast")
    """

    CATEGORY_TO_LABEL = {"tops": 1, "bottoms": 2, "one-pieces": 3}

    def __init__(
        self,
        weights_dir: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        low_vram: Optional[bool] = None,
        compile_model: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.weights_dir = os.path.abspath(weights_dir)
        self.logger = logger or setup_logger("KAPADIYA.AI", level=logging.INFO)

        # Setup device
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger.info(f"Using device: {self.device}")

        if self.device.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            self.logger.info(f"GPU: {torch.cuda.get_device_name(self.device)} ({vram_gb:.1f} GB)")
            if low_vram is None:
                low_vram = vram_gb < LOW_VRAM_THRESHOLD_GB
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.low_vram = bool(low_vram) and self.device.type == "cuda"
        self.logger.info(f"Low-VRAM mode: {self.low_vram}")

        # Setup inference dtype
        self.inference_dtype = self._resolve_dtype(dtype)
        self.logger.info(f"Using dtype: {self.inference_dtype}")

        # Validate weights exist
        self._validate_weights()

        # Load models
        self._setup_tryon_model(compile_model)
        self._setup_pose_model()
        self._setup_hp_model()

        # Pose/parsing run at the native resolution; the model input size depends on the chosen resolution.
        h, w = RESOLUTIONS["full"]
        self.pre_resize = AspectPreserveResize(target_size=(max(h, w), max(h, w)), mode="fit", backend="pil")
        self.resize_pad_fns = {name: ResizePad((rw, rh), backend="opencv") for name, (rh, rw) in RESOLUTIONS.items()}

        self._garment_cache: "OrderedDict[tuple, Dict[str, torch.Tensor]]" = OrderedDict()
        self.garment_cache_size = 8

    def _resolve_dtype(self, dtype: Optional[str]) -> torch.dtype:
        if dtype is not None:
            if dtype not in _DTYPES:
                raise ValueError(f"dtype must be one of {list(_DTYPES)}, got {dtype!r}")
            return _DTYPES[dtype]
        if self.device.type != "cuda":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # Pre-Ampere GPUs (GTX 16xx / RTX 20xx): fp32 would not fit in 4 GB, fp16 does.
        return torch.float16

    def _validate_weights(self):
        """Check that required weight files exist."""
        tryon_path = os.path.join(self.weights_dir, "model.safetensors")
        dwpose_dir = os.path.join(self.weights_dir, "dwpose")
        yolox_path = os.path.join(dwpose_dir, "yolox_l.onnx")
        dwpose_path = os.path.join(dwpose_dir, "dw-ll_ucoco_384.onnx")

        missing = [p for p in (tryon_path, yolox_path, dwpose_path) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing model weights:\n"
                + "\n".join(f"  - {p}" for p in missing)
                + f"\n\nPlease run:\n  python scripts/download_weights.py --weights-dir {self.weights_dir}"
            )

    def _setup_tryon_model(self, compile_model: bool):
        """Load the TryOn model without ever materializing a random-init or fp32 copy."""
        model_path = os.path.join(self.weights_dir, "model.safetensors")
        self.logger.info(f"Loading {model_path}")

        with torch.device("meta"):
            model = TryOnModel()
        state_dict = load_safetensors_streaming(model_path, device=self.device, dtype=self.inference_dtype)
        model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        self.tryon_model = model.eval().requires_grad_(False)
        self._forward = self.tryon_model.forward

        if compile_model:
            try:
                self._forward = torch.compile(self.tryon_model.forward, mode="max-autotune-no-cudagraphs", dynamic=False)
                self.logger.info("torch.compile enabled")
            except Exception as e:  # Triton missing (e.g. Windows) or unsupported backend
                self.logger.warning(f"torch.compile unavailable, running eager: {e}")

        n_params = sum(p.numel() for p in self.tryon_model.parameters()) / 1e9
        self.logger.info(f"TryOnModel loaded ({n_params:.2f}B params)")

    def _setup_pose_model(self):
        """Load DWPose model."""
        dwpose_dir = os.path.join(self.weights_dir, "dwpose")
        self.logger.info(f"Loading DWPose from {dwpose_dir}")

        on_gpu = self.device.type == "cuda" and not self.low_vram
        dwpose_device = f"cuda:{self.device.index or 0}" if on_gpu else "cpu"
        self.pose_model = DWposeDetector(checkpoints_dir=dwpose_dir, device=dwpose_device)

        self.logger.info(f"DWPose loaded ({dwpose_device})")

    def _setup_hp_model(self):
        """Load human parsing model."""
        hp_device = "cuda" if self.device.type == "cuda" and not self.low_vram else "cpu"
        self.hp_model = FashnHumanParser(device=hp_device)
        self.logger.info(f"FashnHumanParser loaded ({hp_device})")

    # ------------------------------------------------------------------ preprocessing

    def _to_tensor(self, img: np.ndarray, num_samples: int) -> torch.Tensor:
        t = normalize_uint8_to_neg1_1(numpy_to_torch(img).unsqueeze(0))
        return t.to(self.device, dtype=self.inference_dtype).repeat(num_samples, 1, 1, 1)

    @staticmethod
    def _labels_for(category: str) -> Tuple[str, List[int]]:
        body_coverage = CATEGORY_TO_BODY_COVERAGE.get(category)
        labels = BODY_COVERAGE_TO_FASHN_LABELS.get(body_coverage)
        return body_coverage, [FASHN_LABELS_TO_IDS[label] for label in labels]

    def _prepare_garment(
        self, garment_image: Image.Image, category: str, garment_photo_type: str, resolution: str
    ) -> Dict[str, torch.Tensor]:
        """Pose + parse + mask the garment once; later calls with the same garment hit the cache."""
        digest = hashlib.md5(garment_image.tobytes()).hexdigest()
        key = (digest, garment_image.size, category, garment_photo_type, resolution, self.inference_dtype)
        if key in self._garment_cache:
            self._garment_cache.move_to_end(key)
            return self._garment_cache[key]

        garment_image = self.pre_resize(garment_image, allow_upsampling=False)
        garment_np = np.array(garment_image)

        garment_pose = (
            get_dummy_dw_keypoints() if garment_photo_type == "flat-lay" else self.pose_model(garment_np[..., ::-1])
        )
        garment_pose_img = draw_pose(garment_pose, garment_np.shape[0], garment_np.shape[1], grayscale=True)
        garment_seg = self.hp_model.predict(garment_np)

        _, label_ids = self._labels_for(category)
        garment_processed = create_garment_image(
            img_np=garment_np,
            seg_pred=garment_seg,
            labels_to_segment_indices=label_ids,
            disable_masking=garment_photo_type == "flat-lay",
        )

        resize_pad = self.resize_pad_fns[resolution]
        entry = {
            "garment_images": self._to_tensor(resize_pad(garment_processed), 1),
            "garment_poses": self._to_tensor(resize_pad(garment_pose_img, interpolation=cv2.INTER_NEAREST_EXACT), 1),
        }
        self._garment_cache[key] = entry
        while len(self._garment_cache) > self.garment_cache_size:
            self._garment_cache.popitem(last=False)
        return entry

    def _prepare_person(
        self, person_image: Image.Image, category: str, segmentation_free: bool, resolution: str
    ) -> Dict[str, torch.Tensor]:
        person_image = self.pre_resize(person_image, allow_upsampling=False)
        person_np = np.array(person_image)

        person_pose = self.pose_model(person_np[..., ::-1])  # DWPose expects BGR
        person_pose_img = draw_pose(person_pose, person_np.shape[0], person_np.shape[1], grayscale=True)
        person_seg = self.hp_model.predict(person_np)

        body_coverage, label_ids = self._labels_for(category)
        ca_image = create_clothing_agnostic_image(
            img_np=person_np.copy(),
            seg_pred=person_seg.copy(),
            labels_to_segment_indices=label_ids,
            body_coverage=body_coverage,
            disable_masking=segmentation_free,
            logger=self.logger,
        )

        resize_pad = self.resize_pad_fns[resolution]
        return {
            "ca_images": self._to_tensor(resize_pad(ca_image, mem_padding=True), 1),
            "person_poses": self._to_tensor(resize_pad(person_pose_img, interpolation=cv2.INTER_NEAREST_EXACT), 1),
        }

    # ------------------------------------------------------------------ sampling

    def _velocity(self, images: torch.Tensor, t_vec: torch.Tensor, cond: dict, mask: Optional[torch.Tensor] = None):
        return self._forward(images, t_vec, mask=mask, **cond)["x"]

    @torch.inference_mode()
    def _sample(
        self,
        *,
        cond: Dict[str, torch.Tensor],
        generator: torch.Generator,
        num_timesteps: int,
        time_shift_mu: float,
        guidance_scale: float,
        cfg_steps: Optional[int],
        skip_cfg_last_n_steps: int,
        init_images: Optional[torch.Tensor] = None,
        strength: float = 1.0,
        use_tqdm: bool = True,
    ) -> torch.Tensor:
        """Rectified-flow Euler sampling (t: 0=noise -> 1=image) with interval CFG and optional warm start."""
        ref = cond["ca_images"]
        batch_size, device, dtype = ref.shape[0], ref.device, ref.dtype
        shape = (batch_size, self.tryon_model.channels_in, ref.shape[-2], ref.shape[-1])

        timesteps = get_rf_schedule(num_steps=num_timesteps, mu=time_shift_mu)
        images = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)

        start = 0
        if init_images is not None and strength < 1.0:
            # Re-noise the previous result to an intermediate t and only integrate the remaining steps.
            start = min(max(int(round(num_timesteps * (1.0 - strength))), 0), num_timesteps - 1)
            t0 = timesteps[start]
            images = (1.0 - t0) * images + t0 * init_images.to(device=device, dtype=dtype).expand(shape)

        ones = torch.ones(batch_size, device=device, dtype=torch.bool)
        zeros = torch.zeros(batch_size, device=device, dtype=torch.bool)

        steps = range(start, num_timesteps)
        for step_idx in tqdm(steps, desc="Sampling", disable=not use_tqdm):
            t_curr, t_next = timesteps[step_idx], timesteps[step_idx + 1]
            t_vec = torch.full((batch_size,), t_curr, dtype=dtype, device=device)

            use_cfg = (
                guidance_scale != 1.0
                and (cfg_steps is None or step_idx < cfg_steps)
                and not (skip_cfg_last_n_steps > 0 and step_idx >= num_timesteps - skip_cfg_last_n_steps)
            )

            if not use_cfg:
                v = self._velocity(images, t_vec, cond, mask=ones)
            elif self.low_vram:
                v_c = self._velocity(images, t_vec, cond, mask=ones)
                v_u = self._velocity(images, t_vec, cond, mask=zeros)
                v = v_u + guidance_scale * (v_c - v_u)
            else:
                pred = self.tryon_model.forward_for_cfg(images, t_vec, **cond)
                v = pred["v_u"] + guidance_scale * (pred["v_c"] - pred["v_u"])

            images = images + (t_next - t_curr) * v

        return images.float().clamp_(-1.0, 1.0)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def warmup(self, resolution: str = "full"):
        """Run one dummy forward so the first real request does not pay kernel-selection cost."""
        h, w = RESOLUTIONS[resolution]
        c = self.tryon_model.channels_in
        z = lambda ch: torch.zeros((1, ch, h, w), device=self.device, dtype=self.inference_dtype)  # noqa: E731
        cond = {
            "ca_images": z(c),
            "person_poses": z(1),
            "garment_images": z(c),
            "garment_poses": z(1),
            "garment_categories": torch.ones(1, dtype=torch.long, device=self.device),
        }
        t = torch.zeros(1, device=self.device, dtype=self.inference_dtype)
        self._velocity(z(c), t, cond)
        self._sync()

    # ------------------------------------------------------------------ public API

    @torch.inference_mode()
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: Literal["tops", "bottoms", "one-pieces"],
        garment_photo_type: Literal["model", "flat-lay"] = "model",
        preset: Optional[Literal["live", "fast", "balanced", "quality"]] = None,
        num_samples: int = 1,
        num_timesteps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        cfg_steps: Optional[int] = -1,
        skip_cfg_last_n_steps: Optional[int] = None,
        resolution: Optional[Literal["full", "preview"]] = None,
        time_shift_mu: float = 1.5,
        seed: int = 42,
        segmentation_free: bool = True,
        init_image: Optional[torch.Tensor] = None,
        strength: float = 1.0,
        use_tqdm: bool = True,
    ) -> PipelineOutput:
        """
        Run virtual try-on inference.

        Args:
            person_image: RGB image of the person to dress.
            garment_image: RGB image of the garment (model photo or flat-lay).
            category: "tops", "bottoms", or "one-pieces".
            garment_photo_type: "model" if worn by a person, "flat-lay" for product shots.
            preset: "live" | "fast" | "balanced" | "quality". Explicit arguments override it.
                Default when nothing is set: "balanced" (identical to upstream FASHN VTON v1.5).
            num_samples: Number of output images to generate (1-4).
            num_timesteps: Euler steps. Fewer = faster.
            guidance_scale: Classifier-free guidance strength.
            cfg_steps: Apply CFG only on the first N steps (None = all steps, -1 = use preset).
                Steps without CFG cost half as much.
            skip_cfg_last_n_steps: Skip CFG for final N steps to prevent color saturation.
            resolution: "full" (864x576) or "preview" (576x384, faster, experimental).
            time_shift_mu: Schedule shift (higher = more time at high noise).
            seed: Random seed for reproducibility.
            segmentation_free: Generate without masking the person image (recommended).
            init_image: `tensors` from a previous output to warm-start from (live/video).
            strength: With init_image, fraction of the schedule to re-run (1.0 = from pure noise).
            use_tqdm: Show a progress bar.

        Returns:
            PipelineOutput with generated `images`, raw `tensors`, and per-stage `timings` (seconds).
        """
        cfg = dict(PRESETS[preset or "balanced"])
        if num_timesteps is not None:
            cfg["num_timesteps"] = num_timesteps
        if guidance_scale is not None:
            cfg["guidance_scale"] = guidance_scale
        if cfg_steps != -1:
            cfg["cfg_steps"] = cfg_steps
        if skip_cfg_last_n_steps is not None:
            cfg["skip_cfg_last_n_steps"] = skip_cfg_last_n_steps
        if resolution is not None:
            cfg["resolution"] = resolution
        resolution = cfg.pop("resolution")
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {list(RESOLUTIONS)}, got {resolution!r}")

        t_start = time.perf_counter()
        np.random.seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        garment = self._prepare_garment(garment_image, category, garment_photo_type, resolution)
        t_garment = time.perf_counter()
        person = self._prepare_person(person_image, category, segmentation_free, resolution)
        t_person = time.perf_counter()

        cond = {**person, **garment}
        cond = {k: v.repeat(num_samples, 1, 1, 1) for k, v in cond.items()}
        cond["garment_categories"] = torch.full(
            (num_samples,), self.CATEGORY_TO_LABEL[category], dtype=torch.long, device=self.device
        )

        if init_image is not None and init_image.shape[-2:] != cond["ca_images"].shape[-2:]:
            init_image = None  # resolution/aspect changed - cannot warm start

        images = self._sample(
            cond=cond,
            generator=generator,
            time_shift_mu=time_shift_mu,
            init_images=init_image,
            strength=strength,
            use_tqdm=use_tqdm,
            **cfg,
        )
        self._sync()
        t_sample = time.perf_counter()

        if not torch.isfinite(images).all():
            self.logger.warning(
                f"Non-finite values in output with dtype {self.inference_dtype}; "
                "if images look broken, retry with dtype='float32' (or 'bfloat16' if your GPU supports it)."
            )
            images = torch.nan_to_num(images)

        resize_pad = self.resize_pad_fns[resolution]
        pil_images = [resize_pad.unpad(tensor_to_pil(img, unnormalize=True)) for img in images]
        t_end = time.perf_counter()

        timings = {
            "garment_preprocess": t_garment - t_start,
            "person_preprocess": t_person - t_garment,
            "sampling": t_sample - t_person,
            "postprocess": t_end - t_sample,
            "total": t_end - t_start,
        }
        self.logger.debug(" | ".join(f"{k}={v:.2f}s" for k, v in timings.items()))
        return PipelineOutput(images=pil_images, tensors=images, timings=timings)
