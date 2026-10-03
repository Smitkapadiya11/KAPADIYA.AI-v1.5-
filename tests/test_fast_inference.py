"""
Tests for the KAPADIYA.AI v1.5 TMKB inference optimizations, using a tiny random model
so they run on CPU in seconds without downloading weights.
"""

import logging

import pytest
import torch
from safetensors.torch import save_file

from kapadiya_ai.pipeline import TryOnPipeline
from kapadiya_ai.tryon_mmdit import TryOnModel
from kapadiya_ai.utils import load_safetensors_streaming

TINY = dict(
    input_shape=(864, 576),
    hidden_size=128,
    n_heads=2,
    double_blocks_depth=1,
    single_blocks_depth=1,
    patch_mixer_depth=1,
    axes_dim=(16, 24, 24),
)


def _cond(h, w, batch=1):
    g = torch.Generator().manual_seed(0)
    r = lambda c: torch.rand((batch, c, h, w), generator=g) * 2 - 1  # noqa: E731
    return {
        "ca_images": r(3),
        "person_poses": r(1),
        "garment_images": r(3),
        "garment_poses": r(1),
        "garment_categories": torch.ones(batch, dtype=torch.long),
    }


def _pipe(model, low_vram=False):
    """Build a pipeline around an in-memory model, skipping weight files and aux models."""
    pipe = TryOnPipeline.__new__(TryOnPipeline)
    pipe.tryon_model = model.eval()
    pipe._forward = model.forward
    pipe.low_vram = low_vram
    pipe.device = torch.device("cpu")
    pipe.inference_dtype = torch.float32
    pipe.logger = logging.getLogger("test")
    return pipe


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = TryOnModel(**TINY)
    # Zero-initialised gates would make every block an identity; randomise so tests are meaningful.
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.02)
    return m.eval()


def test_streaming_meta_load_matches_original(model, tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))

    with torch.device("meta"):
        loaded = TryOnModel(**TINY)
    sd = load_safetensors_streaming(str(path), device=torch.device("cpu"), dtype=torch.float32)
    loaded.load_state_dict(sd, strict=True, assign=True)

    assert not any(p.is_meta for p in loaded.parameters())
    assert not any(b.is_meta for b in loaded.buffers())
    for (k, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), k


@pytest.mark.parametrize("hw", [(864, 576), (576, 384)])
def test_forward_is_resolution_agnostic(model, hw):
    h, w = hw
    x = torch.randn(1, 3, h, w)
    with torch.inference_mode():
        out = model(x, torch.zeros(1), **_cond(h, w))["x"]
    assert out.shape == (1, 3, h, w)


def test_cached_rope_matches_per_batch_rope(model):
    """Batched (CFG) forward with the cached batch-1 RoPE must equal per-sample forwards."""
    h, w = 576, 384
    x = torch.randn(2, 3, h, w)
    cond = _cond(h, w, batch=2)
    with torch.inference_mode():
        both = model(x, torch.full((2,), 0.3), **cond)["x"]
        single = torch.cat(
            [model(x[i : i + 1], torch.full((1,), 0.3), **{k: v[i : i + 1] for k, v in cond.items()})["x"] for i in range(2)]
        )
    torch.testing.assert_close(both, single, atol=1e-4, rtol=1e-4)


def test_low_vram_sequential_cfg_matches_batched_cfg(model):
    h, w = 576, 384
    kw = dict(num_timesteps=4, time_shift_mu=1.5, guidance_scale=1.5, cfg_steps=None, skip_cfg_last_n_steps=1)
    a = _pipe(model, low_vram=False)._sample(cond=_cond(h, w), generator=torch.Generator().manual_seed(1), use_tqdm=False, **kw)
    b = _pipe(model, low_vram=True)._sample(cond=_cond(h, w), generator=torch.Generator().manual_seed(1), use_tqdm=False, **kw)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    assert a.shape == (1, 3, h, w)


def _count_forwards(model, **kw):
    calls = {"n": 0}
    orig = model.forward

    def counting(*a, **k):
        calls["n"] += a[0].shape[0]  # count per-sample passes (CFG batches 2)
        return orig(*a, **k)

    pipe = _pipe(model, low_vram=kw.pop("low_vram", False))
    pipe._forward = counting
    model.forward = counting
    try:
        pipe._sample(cond=_cond(576, 384), generator=torch.Generator().manual_seed(1), use_tqdm=False, **kw)
    finally:
        del model.forward
    return calls["n"]


def test_cfg_interval_reduces_model_passes(model):
    base = dict(num_timesteps=8, time_shift_mu=1.5, guidance_scale=1.5, skip_cfg_last_n_steps=1)
    full_cfg = _count_forwards(model, cfg_steps=None, **base)
    interval = _count_forwards(model, cfg_steps=3, **base)
    assert full_cfg == 7 * 2 + 1  # upstream behaviour: CFG on all but the last step
    assert interval == 3 * 2 + 5


def test_warm_start_runs_fewer_steps(model):
    init = torch.zeros(1, 3, 576, 384)
    kw = dict(num_timesteps=8, time_shift_mu=1.5, guidance_scale=1.0, cfg_steps=None, skip_cfg_last_n_steps=0)
    assert _count_forwards(model, **kw) == 8
    assert _count_forwards(model, init_images=init, strength=0.5, **kw) == 4
