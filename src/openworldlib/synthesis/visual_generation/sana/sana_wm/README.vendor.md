# Sana-WM Vendor Code

## Origin

- **Repository**: https://github.com/NVlabs/Sana
- **Branch**: main
- **Baseline Commit**: `40151c81172a981f4ba24286cf9ef4b08912f653`
- **Commit Message**: "Add SANA-WM initial release (#379)"
- **Date**: 2025-07-16

## What's Included

This directory (`sana_wm_diffusion/`) is a **cropped vendor copy** of Sana's upstream `diffusion/` package, renamed to `sana_wm_diffusion` to avoid package name collision with both `diffusers` and OpenWorldLib's own `base_models.diffusion_model`.

### Core components kept

| Component | Path | Description |
|-----------|------|-------------|
| Sana DiT Models | `model/nets/sana_multi_scale_video_camctrl.py` | Sana-WM camera-controlled DiT backbone |
| Sana Blocks | `model/nets/sana_blocks.py`, `sana_camctrl_blocks.py` | Attention + Cross-Attention blocks |
| GDN Blocks | `model/nets/sana_gdn_blocks*.py` | Gated DeltaNet blocks |
| Model Registry | `model/builder.py`, `model/registry.py` | Model build + registry (mmcv-based) |
| Schedulers | `scheduler/flow_euler_sampler.py` | `LTXFlowEuler`, `FlowEuler` |
| Camera Utils | `utils/cam_utils.py`, `utils/chunk_utils.py` | Camera conditioning tools | |
| Triton Ops | `model/ops/fused_gdn*.py` | GPU kernel fused GDN |
| Refiner | `refiner/diffusers_ltx2_refiner.py` | LTX-2 stage-2 refiner wrapper |
| Wan Support | `model/wan/model.py`, `clip.py`, `vae.py` | Wan-1 text encoder + VAE wrappers |

### Additional copied files

- `sana/tools/` — Download helpers (`hf_utils.py`, `download.py`)
- `tools/download.py` — Checkpoint loading (`find_model()`)

## What's Removed (vs upstream)

All training-only, data-processing, and non-inference components:

| Removed | Reason |
|---------|--------|
| `data/` | Training data pipeline |
| `longsana/` | Long-context training |
| `post_training/` | Post-training pipeline |
| `model/dc_ae/` | DC-AE VAE (unused in Sana-WM) |
| `model/qwen/` | Qwen text encoder (unused) |
| `model/wan2_2/` | Wan2.2 VAE (unused) |
| `model/nets/fastlinear/` | Triton linear ops (unused) |
| `model/wan/attention.py`, `t5.py`, `tokenizers.py`, etc. | Wan training-only files |

## Import Renames

All imports in the copied files have been rewritten as part of vendoring:

- `diffusion.XXX` → `sana_wm_diffusion.XXX`
- All relative intra-package imports (`from .sana import ...`) preserved intact

## License

Apache 2.0 — same as upstream. See `LICENSE` at root of OpenWorldLib for full terms.