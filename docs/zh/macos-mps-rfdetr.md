# RF-DETR v6 Apple/MPS inference（Issue #8）

## 範圍及架構審查

Base：`c8697c9`（`feature/macos-mps`）；前置 #2/#3/#4/#5/#7 已合併。

`RestorationSession.detection_model_for`／`build_compiled_detection_model` →
`detection_registry.build_detection_model` → `RfDetrMosaicDetectionModel` →
`RfDetrTorchRunner` → RF-DETR LWDETR。`pipeline_processing.process_batch`／tracker
消費 CPU float32 pixel boxes 與 GPU bool masks；GUI scan 消費 GPU float32 scores
與 GPU bool merged masks。保持這些介面，沒有新增 backend abstraction。

Apple 與 AMD 共用 Torch runner；Apple 使用真正 MPS device、FP32，忽略
`fp16=True` 並記錄訊息。Checkpoint 仍先在 CPU 載入，再移至 MPS。
既有 registry 的 APPLE `.pt` 選擇及 medium 576／large 768 variant 不變；
legacy `torch_variant=None` 明確報錯，不猜模型架構。

NVIDIA 仍走 ONNX → TensorRT 及原有 cache／compiler；AMD 保持原有 autocast。
Apple 不探查 TensorRT cache、不建立 ONNX/TensorRT engine、不執行 CUDA fatbin。
直接呼叫 compile helper 時只驗證 `.pt`；missing checkpoint 訊息提示 explicit path
或 `JASNA_MODEL_WEIGHTS_DIR`。沒有 Windows/Linux guard 或 accelerator API 變更。

已讀 RF-DETR 1.8.3 的 `MSDeformAttn`、`ms_deform_attn_core_pytorch`、
`_bilinear_grid_sample`、DINOv2 attention／positional interpolation、checkpoint loader；
保留 dependency 的 MPS gather-based grid 及 antialias guard。亦對照 upstream
`Kruk2/jasna` 的 AMD Torch runner；不修改第三方套件。

## 已實測限制及 CPU 邊界

- `scan_scores_masks` 的 `area` resize 實測在 MPS 非整除尺寸失敗：
  `Adaptive pool MPS: input sizes must be divisible by output sizes`。
  只把已合併的 `B×1×Hm×Wm` float32 小 mask 移到 CPU 做 area resize，再傳回 MPS。
  可整除尺寸繼續在 MPS；不改成 nearest／bilinear，保留 any-pixel mask semantics。
- 原有 boxes／valid flags → CPU 是既有 `Detections` contract。
  模型、normalize、attention、query gather／top-k、selected bool masks 都在 MPS。
- 沒有設定 `PYTORCH_ENABLE_MPS_FALLBACK=1`，沒有整個模型 CPU fallback。
  測試中的完整 CPU model inference 只是 reference。
- Apple FP16/BF16 未支援／未驗證；本次只驗證 FP32。
- `rfdetr-vr-v1.pt` large/768 已獨立完成真實 batch 1/2 inference 與 scan；
  一般平面 fixture 沒有 VR 正偵測，不能聲稱 VR detection quality／完整 VR pipeline 支援。
  既有 advanced-video MPS gate 保留。沒有 `rfdetr-v6-large.pt`，不聲稱該 weights 已驗證。
- 不包含 restoration、threaded video I/O、CLI E2E 或後續 issue implementation。

## CPU reference 診斷

v6 fixture frames 0/120：production threshold 0.35 下 CPU/MPS counts 都是 `[0, 1]`；
selected mask 完全相同，pixel boxes 最大誤差約 0.00348；測試要求 boxes ≤0.05 pixel、
best score ≤1e-4、selected bool masks 完全相同。另檢查全部 v6 normalized boxes
誤差 ≤1e-3、sigmoid class probabilities 誤差 ≤1e-3，以及 raw tensors 的 finite/dtype/device。

初次探索把所有 raw mask logits 逐項要求 `atol=rtol=0.003`，未通過。
v6 raw max errors（boxes/logits/masks）為 0.000800/0.004616/0.135761；
mask-logit RMSE 約 0.003205，全部 mask sign agreement 約 0.9999967。
原始 query logits 不是對外 binary mask contract，因此正式 reference 檢查 selected masks，
並保留上述原始差異作診斷，沒有改動或削弱任何既有測試。

VR raw query-by-index 差異較大。追蹤 encoder proposal top-k：CPU/MPS logits 最大差異
僅 5.70e-5，但 frame 0 的 query 111/112（proposal 3081/3563）交換排序，
近似 logits 約 -4.12368。背景 query indices 不保證跨 backend 對齊；
不能把 raw index-wise errors 宣稱為精度相等。最終 counts 都是 `[0, 0]`。

獨立真實 op probes：RF-DETR gather grid 對 CPU 最大誤差 ≤4.77e-7；
SDPA ≤5.37e-7；LayerNorm ≤4.77e-7。正式 tests 包含 zeros/border、align_corners
兩種設定、long gather／bool indexing、空／非空 detection、scan 下採樣／上採樣。

## 環境及唯讀模型

2026-10-10，Apple M2 Pro，32 GiB unified memory，arm64；macOS 27.0.1 / 26A434。
Python 3.12.4 / 3.13.13；兩環境皆 Torch 2.12.0、torchvision 0.27.0、
rfdetr 1.8.3、transformers 5.1.0、numpy 2.5.3、PyAV 18.1.0。
系統 FFmpeg/ffprobe 8.0.1。

真實主機 `is_built()/is_available()` 都是 `True/True`；sandbox 內為 `True/False`，
GPU tests 使用本機非 sandbox 執行，沒有把 unavailable 當作 skip。

模型目錄唯讀：`/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights`。
驗證前後 SHA256 相同：

| 模型 | SHA256 |
|---|---|
| rfdetr-v6.pt | f10bedc4d105c2721e4259b8680203d51f344f73e55e85710d915619f5731b55 |
| rfdetr-vr-v1.pt | 55543c83911921ef79cd8cae8540bd25e34c7daf488e77f79d233d6926973a2e |

## 可重現命令及結果

於 repository root 執行。這兩個既有虛擬環境重用 #2 的已安裝 dependencies：

```bash
PY312=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
PY313=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-313/bin/python
WEIGHTS='/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights'
EVIDENCE=/Users/kaho/jasna-mac/verification-evidence/issue8

"$PY312" --version
"$PY313" --version
"$PY312" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize
"$PY312" -c 'import importlib.metadata as m; print({p:m.version(p) for p in ("torch","rfdetr","transformers","torchvision","av","numpy")})'

"$PY312" -m pytest -q tests/test_rfdetr_preprocess.py tests/test_rfdetr_postprocess.py tests/test_detection_registry.py
# 61 passed
"$PY312" -m pytest -q tests/test_rfdetr_preprocess.py tests/test_rfdetr_postprocess.py tests/test_detection_registry.py tests/test_rfdetr_mps.py
# 73 passed

JASNA_TEST_MODEL_WEIGHTS_DIR="$WEIGHTS" "$PY312" -m pytest -q -s tests/test_mps_rfdetr_inference.py
JASNA_TEST_MODEL_WEIGHTS_DIR="$WEIGHTS" "$PY313" -m pytest -q -s tests/test_mps_rfdetr_inference.py
# 8 passed each; no skips

"$PY312" -m pytest -q tests/test_rfdetr_preprocess.py tests/test_rfdetr_postprocess.py tests/test_detection_registry.py tests/test_rfdetr_mps.py tests/test_amd_support.py tests/test_mps_model_loading.py tests/test_session_factory.py tests/test_engine_compiler.py
"$PY313" -m pytest -q tests/test_rfdetr_preprocess.py tests/test_rfdetr_postprocess.py tests/test_detection_registry.py tests/test_rfdetr_mps.py tests/test_amd_support.py tests/test_mps_model_loading.py tests/test_session_factory.py tests/test_engine_compiler.py
# 147 passed, 5 skipped each (opt-in weights tests / platform case)

"$PY312" -m scripts.verify_mps_rfdetr_video --weights-dir "$WEIGHTS" "$EVIDENCE/v6-detections.mp4"
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,width,height,nb_frames,nb_read_frames,duration,avg_frame_rate -of json "$EVIDENCE/v6-detections.mp4"
ffmpeg -v error -i "$EVIDENCE/v6-detections.mp4" -f null -
# 8 frames, 2 positive detections; H.264 960x540, 30 fps, 0.266667 s;
# nb_frames=nb_read_frames=8, monotonic PTS, full decode succeeds

shasum -a 256 "$WEIGHTS/rfdetr-v6.pt" "$WEIGHTS/rfdetr-vr-v1.pt"
git diff --check
```

診斷影片使用 PyAV decode → 真實 MPS detector → Pillow overlay → PyAV libx264。
CPU overlay/encode 是輸出邊界；沒有 restoration；診斷輸出不含 audio。
MPS v6 batch2 耗時約 0.24 秒（暖機後）；包括 detection postprocess／host boxes，
不包括 decode／overlay／encode，不能當完整 pipeline FPS benchmark。

## Regression 與風險

Focused NVIDIA compiler/runner mock contract、AMD routing／autocast、engine compiler、
session factory regression 通過；本機無 NVIDIA/AMD hardware，未聲稱實機驗證。

額外 `tests/test_accelerator_rocm_env.py`：1 failed / 3 passed，
因 `ROCM_DEFAULTS` 測試漏列既有 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`。
以 `git archive origin/feature/macos-mps` 到 `/tmp/jasna-issue8-base`，再執行：

```bash
cd /tmp
PYTHONPATH=/tmp/jasna-issue8-base /Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python -m pytest -q /tmp/jasna-issue8-base/tests/test_accelerator_rocm_env.py
# 同樣 1 failed / 3 passed；本 Issue 不改 ROCm defaults 或其既有 assertion
```

RF-DETR 有 `FutureWarning`（deprecated decorator）、DINOv2 positional/patch-size
提示及 trimmed checkpoint 缺 args.num_queries（由 2600/13 正確推導 200）提示。
完整 checkpoint 載入並完成 inference；沒有為此下載或替換來源 weights。
Full suite 未執行；本交付只覆蓋本 Issue 及相關 regression。
