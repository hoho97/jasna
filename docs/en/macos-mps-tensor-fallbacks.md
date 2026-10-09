# MPS tensor fallbacks (issue #7)

Scope: the six CUDA fatbins and their existing Torch equivalents, based on
`c234609` (`origin/feature/macos-mps`). Dependency #3 is already merged via PR #20.
This does not enable the complete macOS video/restoration pipeline.

## Architecture and caller/callee review

All six families select their kernels using `is_nvidia_device(device)` from
`accelerator.py`. `Kernel` construction is lazy: only `launch` resolves a fatbin
and reaches `cuInit` / `cuLaunchKernel`. MPS must never reach that boundary.
There is no new Metal kernel, CUDA emulation, global CPU fallback, or changed
CUDA driver/platform loader. The existing ROCm eager paths are retained.

| Family | Caller / Torch callee on Apple | Policy |
| --- | --- | --- |
| resize_normalize | YOLO / RF-DETR `_preprocess` → `F.interpolate`, normalization / letterbox | `ResizeNormalizer.available` stays false; callers use their existing Torch expressions. Tested FP32/FP16, odd source sizes and non-unit source pixel strides. |
| yuv_to_rgb | decoder plane conversion → `_convert_eager` | MPS NV12 uint8 and P010 uint16 planes, FP32 reusable scratch, ordered 10-bit dither, uint8 CHW output. Even 4:2:0 dimensions required. |
| rgb_to_yuv | encoder `_to_yuv` → `_rgb_to_yuv_into` / `YuvScratch` | MPS NV12 uint8 / P010 int16 storage; P010 writes via uint16 views to avoid MPS float→int16 saturation. |
| cas | encoder `_to_yuv` → `sharpen_into` / `_sharpen_torch_` | Existing banded Torch CAS; 8/10-bit luma tested at strengths 0.1/0.5/1.0; chroma preserved. |
| denoise | `RestorationPipeline` → `apply_denoise` / `apply_denoise_u8` → `spatial_denoise` | Existing bilateral Torch loop; spatial weights stay on-device instead of synchronizing each tap with `float(tensor)`. FP32/FP16 tested. |
| lut | encoder `_encode_frame` → `GpuLutApplier.apply` → `_apply_1d` / `_apply_3d` | Existing Torch indexing and 5D `grid_sample`; non-identity 1D/3D LUTs, custom domains, uint8/FP32 and non-unit pixel strides tested. |

YUV eager and NVIDIA paths now check plane/output shapes and common device.
MPS additionally checks exact plane dtypes. CPU floating-point reference inputs
remain accepted. Raw CUDA surface/frame APIs continue to reject MPS with their
existing explicit `requires a CUDA converter` messages. RGB input still requires
uint8 and contiguous pixels (pitched rows supported). Odd 4:2:0 dimensions are
rejected; detector resize, LUT and denoise accept odd spatial dimensions.

The `.cu` implementations, shared scratch math and relevant caller branches were
reviewed. NVIDIA kernels and launch arguments are unchanged. Windows/Linux
platform loading and ROCm selection are unchanged. Portable dispatch tests cover
both NVIDIA and non-NVIDIA branch selection, but do not substitute for real
NVIDIA/AMD hardware regression tests.

## Real Apple Silicon verification (2026-10-10)

Apple M2 Pro, 32 GiB unified memory, arm64, macOS 27.0.1 (26A434).
Python 3.12.4 and 3.13.13, Torch 2.12.0, torchvision 0.27.0,
PyAV 18.1.0, NumPy 2.5.3, pytest 9.1.1. MPS built/available = true/true.
System FFmpeg/ffprobe 8.0.1. PyAV bundled libavcodec 62.28.102 /
libavformat 62.12.102 / libswscale 9.5.102.

Commands below ran from the repository. Existing dependency environments were
reused; no dependencies or model weights were modified.

```sh
PY312=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
PY313=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-313/bin/python
$PY312 --version
$PY312 -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize

PYTORCH_ENABLE_MPS_FALLBACK=0 $PY312 -m pytest -q tests/test_resize_normalize.py tests/test_yuv_to_rgb.py tests/test_rgb_to_nv12.py tests/test_cas.py tests/test_lut.py tests/test_denoise.py
# 73 passed, 19 skipped (CUDA-only).
PYTORCH_ENABLE_MPS_FALLBACK=0 $PY312 -m pytest -q tests/test_mps_tensor_fallbacks.py tests/test_tensor_kernel_dispatch.py
# 41 passed (39 actual MPS cases, 2 portable dispatch cases).

PYTORCH_ENABLE_MPS_FALLBACK=0 $PY312 -m pytest -q tests/test_mps_tensor_fallbacks.py tests/test_tensor_kernel_dispatch.py tests/test_resize_normalize.py tests/test_yuv_to_rgb.py tests/test_rgb_to_nv12.py tests/test_cas.py tests/test_lut.py tests/test_denoise.py tests/test_rgb_to_p010.py tests/test_amd_support.py tests/test_mps_accelerator.py tests/test_denoise_kernel.py tests/test_lut_kernel.py tests/test_rgb_to_yuv_kernel.py tests/test_yuv_scratch_reuse.py tests/test_rfdetr_preprocess.py tests/test_yolo_letterbox.py tests/test_no_cpu_tensor_ops.py
# Repeat with $PY313; 171 passed, 88 skipped on each Python.

PYTORCH_ENABLE_MPS_FALLBACK=0 $PY312 -m scripts.verify_mps_tensor_video /tmp/jasna-issue7-mps.mp4
ffprobe -v error -count_frames -show_entries stream=codec_name,width,height,nb_read_frames,r_frame_rate,duration,color_range,color_space -of json /tmp/jasna-issue7-mps.mp4
ffmpeg -v error -i /tmp/jasna-issue7-mps.mp4 -f null -
```

Video smoke uses `assets/test_clip1_1080p.mp4`: PyAV decode/resize to 320×180
NV12, real MPS YUV→RGB→NV12, explicit host transfer and libx264 software encode.
CPU RGB reference error was zero. Output: H.264, 320×180, limited BT.709,
24 fps, 24 frames, 1.000000 s. PyAV verified increasing PTS and decoded all
frames; FFmpeg also decoded the complete output without errors. MPS conversions
and transfers: median 1.551 ms, first frame 139.869 ms (single short run,
not an end-to-end benchmark). No weights or audio are used in this smoke.

Initial real-MPS tests exposed float→int16 P010 saturation (values above 32767
were clipped). An isolated uint16 view/copy probe and the converter regression
suite verified the GPU-only fix. Odd YUV dimensions initially lacked a clear
error; they now fail at construction.

## Limits and known baseline failure

No operation-level CPU fallback was needed. CPU boundaries are reference
comparisons and explicit software video I/O only. Tests ran with global MPS CPU
fallback disabled. The real-MPS suite fails if any converter reaches the CUDA
driver, a kernel launch or `torch.cuda.current_stream`.

Optional CAS, denoise and LUT are validated here as tensor operations. This does
not claim full-resolution HDR color management, all LUT sizes/dtypes, a memory
stress benchmark, model inference, audio muxing, or threaded video integration.
VideoReader's software staging/stream assumptions and VideoEncoder's Apple
handoff remain issues #10/#11; model runners remain #8/#9. No model weights were
required or accessed. No NVIDIA/AMD GPU is attached; hardware-only tests skip.

An extra regression run of `tests/test_accelerator_rocm_env.py` gives 1 failed /
3 passed because its expected defaults omit the already-existing
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. This was reproduced from unchanged
`c234609` files in `/tmp/jasna-issue7-base`, independently of these edits:

```sh
PYTHONPATH=/tmp/jasna-issue7-base /Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python -m pytest -q /tmp/jasna-issue7-base/tests/test_accelerator_rocm_env.py
```

That unrelated baseline assertion is reported, not weakened or changed in #7.
