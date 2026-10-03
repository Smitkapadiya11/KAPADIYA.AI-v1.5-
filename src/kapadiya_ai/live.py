#!/usr/bin/env python3
"""
KAPADIYA.AI v1.5 TMKB - live webcam try-on.

The camera feed is shown at full frame rate; a background worker continuously
re-dresses the newest frame and the result panel updates as soon as each one is
ready. Successive frames are warm-started from the previous result (re-noised to
`--strength`), which both shortens sampling and keeps the garment stable between
frames instead of flickering.

Keys:  q / Esc  quit     s  save current result     w  toggle warm start
       c  cycle category (tops -> bottoms -> one-pieces)
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import MODEL_NAME
from .pipeline import PRESETS, RESOLUTIONS, TryOnPipeline

CATEGORIES = ["tops", "bottoms", "one-pieces"]


class LiveTryOn:
    def __init__(self, pipeline: TryOnPipeline, garment: Image.Image, args):
        self.pipeline = pipeline
        self.garment = garment
        self.args = args
        self.category = args.category
        self.warm_start = not args.no_warm_start

        self._lock = threading.Lock()
        self._latest_frame = None  # newest camera frame (BGR)
        self._result = None  # newest try-on (BGR)
        self._prev_tensor = None
        self._latency = 0.0
        self._stop = threading.Event()

    def submit(self, frame_bgr: np.ndarray):
        with self._lock:
            self._latest_frame = frame_bgr

    def snapshot(self):
        with self._lock:
            return self._result, self._latency

    def stop(self):
        self._stop.set()

    def cycle_category(self):
        self.category = CATEGORIES[(CATEGORIES.index(self.category) + 1) % len(CATEGORIES)]
        self._prev_tensor = None
        return self.category

    def run(self):
        while not self._stop.is_set():
            with self._lock:
                frame, self._latest_frame = self._latest_frame, None
            if frame is None:
                time.sleep(0.005)
                continue

            person = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t0 = time.perf_counter()
            try:
                out = self.pipeline(
                    person_image=person,
                    garment_image=self.garment,
                    category=self.category,
                    garment_photo_type=self.args.garment_photo_type,
                    preset=self.args.preset,
                    resolution=self.args.resolution,
                    num_timesteps=self.args.num_timesteps,
                    seed=self.args.seed,  # fixed noise across frames = less flicker
                    init_image=self._prev_tensor if self.warm_start else None,
                    strength=self.args.strength,
                    use_tqdm=False,
                )
            except Exception as e:  # e.g. no person detected in this frame
                self.pipeline.logger.warning(f"Frame skipped: {e}")
                continue

            self._prev_tensor = out.tensors[:1]
            result = cv2.cvtColor(np.array(out.images[0]), cv2.COLOR_RGB2BGR)
            with self._lock:
                self._result = result
                self._latency = time.perf_counter() - t0


def _fit(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * height / h)), height), interpolation=cv2.INTER_AREA)


def _label(img: np.ndarray, text: str, y: int):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} - live webcam try-on")
    parser.add_argument("--weights-dir", type=str, required=True)
    parser.add_argument("--garment-image", type=str, required=True)
    parser.add_argument("--category", type=str, choices=CATEGORIES, default="tops")
    parser.add_argument("--garment-photo-type", type=str, choices=["model", "flat-lay"], default="flat-lay")
    parser.add_argument("--preset", type=str, choices=list(PRESETS), default="live")
    parser.add_argument("--resolution", type=str, choices=list(RESOLUTIONS), default=None)
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--strength", type=float, default=0.6, help="Warm-start strength (1.0 = full from noise)")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--low-vram", action="store_true", default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/live")
    args = parser.parse_args()

    garment = Image.open(args.garment_image).convert("RGB")
    pipeline = TryOnPipeline(weights_dir=args.weights_dir, device=args.device, dtype=args.dtype, low_vram=args.low_vram)
    pipeline.warmup(args.resolution or PRESETS[args.preset]["resolution"])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    live = LiveTryOn(pipeline, garment, args)
    worker = threading.Thread(target=live.run, daemon=True)
    worker.start()

    out_dir = Path(args.output_dir)
    window = MODEL_NAME
    print(f"{MODEL_NAME} live - q: quit, s: save, w: warm start on/off, c: cycle category")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            live.submit(frame)

            result, latency = live.snapshot()
            cam_panel = _fit(frame, 540)
            res_panel = _fit(result, 540) if result is not None else np.zeros((540, 360, 3), np.uint8)
            canvas = np.hstack([cam_panel, res_panel])
            _label(canvas, f"{MODEL_NAME}  |  {live.category}  |  warm start {'on' if live.warm_start else 'off'}", 25)
            if result is not None:
                _label(canvas, f"try-on latency {latency * 1000:.0f} ms  ({1 / max(latency, 1e-6):.2f} fps)", 50)
            else:
                _label(canvas, "generating first frame...", 50)
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and result is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"live_{int(time.time())}.png"
                cv2.imwrite(str(path), result)
                print(f"Saved {path}")
            if key == ord("w"):
                live.warm_start = not live.warm_start
            if key == ord("c"):
                print(f"Category: {live.cycle_category()}")
    finally:
        live.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
