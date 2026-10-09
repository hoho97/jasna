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
