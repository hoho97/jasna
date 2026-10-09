# macOS Apple Silicon development

This setup targets native Apple Silicon (`arm64`) Macs on Python 3.12 or 3.13.
The `macos` extra installs PyPI builds of PyTorch and torchvision, which use the
Metal Performance Shaders (MPS) backend on supported Macs. It deliberately does
not install CUDA, ROCm, TensorRT, `python_vali`, or `nvidia-vfx`.

The dependency set is intentionally separate from the NVIDIA and AMD extras so
those release paths keep their existing vendor-specific pins.

## Clean environment

Run each supported Python version in a separate clean virtual environment:

```bash
uv venv --python 3.12 .venv-macos-312
uv pip install --python .venv-macos-312/bin/python -e ".[macos,dev]"
uv pip check --python .venv-macos-312/bin/python

uv venv --python 3.13 .venv-macos-313
uv pip install --python .venv-macos-313/bin/python -e ".[macos,dev]"
uv pip check --python .venv-macos-313/bin/python
```

Do not pass the CUDA or ROCm wheel indexes when installing the macOS extra.
The macOS dependency contract currently pins:

- `torch==2.12.0`
- `torchvision==0.27.0`
- `rfdetr==1.8.3`
- `transformers==5.1.0`
- base `av==18.1.0`
- base `mmengine==0.10.7`

## Dependency and MPS smoke test

Run this in both clean environments:

```bash
.venv-macos-312/bin/python - <<'PY'
import platform
import av
import mmengine
import rfdetr
import torch
import torchvision
import transformers
import ultralytics

assert platform.system() == "Darwin"
assert platform.machine() == "arm64"
assert torch.backends.mps.is_built()
assert torch.backends.mps.is_available()
assert torch.version.cuda is None
assert torch.version.hip is None

print("python", platform.python_version())
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("av", av.__version__)
print("mmengine", mmengine.__version__)
print("rfdetr", getattr(rfdetr, "__version__", "import-ok"))
print("transformers", transformers.__version__)
print("ultralytics", ultralytics.__version__)
print("mps", torch.backends.mps.is_available())
PY
```

Repeat with `.venv-macos-313/bin/python` for Python 3.13.

Also confirm that no vendor GPU runtime slipped into the environment:

```bash
uv pip freeze --python .venv-macos-312/bin/python | \
  grep -Ei 'tensorrt|torch-tensorrt|python[_-]vali|nvidia-vfx|rocm' && exit 1 || true
```

## mmengine compatibility patch

`patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` is retained
unchanged. The macOS dependency setup does not apply it automatically because
that would alter the existing NVIDIA/AMD packaging flow. The smoke test above
establishes whether stock `mmengine==0.10.7` imports with the selected macOS
Torch build. Checkpoint-loading behavior that exercises the patch must be
verified with the real BasicVSR++ weights on Apple Silicon before this port is
considered complete.

## Model paths and cache policy (issue #5)

Reuse the free Windows release checkpoints without editing the release folder.
An explicit `--detection-model-path` / `--restoration-model-path` takes precedence
for that model. Alternatively set the directory before launching Jasna:

```bash
export JASNA_MODEL_WEIGHTS_DIR="/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights"
```

Resolution order: explicit per-model path, `JASNA_MODEL_WEIGHTS_DIR`, frozen
executable's adjacent `model_weights`, source checkout's CWD `model_weights`.
The override is shared by the lightweight engine-path helpers, including their
import-time secondary-model constants; set it before importing Jasna.

Apple and AMD select RF-DETR `.pt`; NVIDIA retains `.onnx` → TensorRT. Apple
explicit detector paths require `.pt`, and BasicVSR++ requires `.pth` with a
tensor state_dict. Errors identify missing files, encrypted weights, incompatible
extensions, corrupt payloads, and wrong checkpoint schemas. The seven-file
inventory, hashes, versions and provenance are in
[`assets/THIRD_PARTY_MODELS.md`](../../assets/THIRD_PARTY_MODELS.md).

CPU deserialization is a loading stage, not a model-execution fallback.
RF-DETR constructs on CPU before final `.to(mps)`; BasicVSR++ loads its strict
state_dict on CPU before transferring parameters and buffers. MPS verification
uses FP32. Do not assume YOLO's serialized FP16 model is ready for FP32 input;
convert the loaded model to FP32 before moving it to MPS.

Apple uses eager PyTorch and creates no TensorRT or compiled-model cache.
Engine preflight never probes existing NVIDIA engine files or launches a
compiler on Apple. Explicit `.engine` or `.onnx` detector inputs are rejected;
YOLO's direct loader also rejects these on MPS. Existing NVIDIA engine names
and Windows/Linux suffixes are unchanged. AMD retains its Torch path. Do not
share generated engines or compiled artifacts across vendors; raw free
checkpoints can be shared read-only. Use a separate writable working weights
copy for NVIDIA compilation, which can write beside its raw models.

This foundation does not enable full RF-DETR MPS inference (#8), BasicVSR++
temporal inference (#9), YOLO pipeline support (#14), or CLI E2E (#12).
The real tests below validate loading and checkpoint-backed submodule operations,
not full detector/restorer output. `unet-4x.onnx.enc` remains unsupported;
no private protection code or CPU inference fallback is introduced.

### Real checkpoint loading verification

In a native macOS environment with the `macos,dev` dependencies installed:

```bash
export JASNA_TEST_MODEL_WEIGHTS_DIR="/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights"
python -m pytest -q tests/test_model_weights_dir.py tests/test_detection_registry.py tests/test_ltx_model_files.py tests/test_mps_model_loading.py
shasum -a 256 "$JASNA_TEST_MODEL_WEIGHTS_DIR/"*
```

The real tests skip only when the opt-in path is unset. With the path set they
require available MPS, all four free PyTorch files, correct tensor counts,
shapes, all parameters/buffers on MPS, FP32 floating values, and finite outputs.
They run both RF-DETR class heads, the BasicVSR++ feature extractor and the YOLO
stem using actual checkpoint weights. No checkpoint is saved or modified.

For the six fatbin families, Apple tensor policies, numerical checks and a
converter-only video smoke, see [MPS tensor fallbacks](macos-mps-tensor-fallbacks.md).
