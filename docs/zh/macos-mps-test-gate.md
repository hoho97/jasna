# macOS 測試分類與真實 MPS gate（Issue #13）

本 gate 重用目前的 accelerator、RF-DETR Torch runner、BasicVSR++、PyAV 與 CLI
測試，沒有第二套 inference pipeline，也不改變 NVIDIA／AMD production 行為。
前置 Issue #12 已在 `190c033`（PR #29）合併。

## 分類及 collection

|Marker|實際需求|
|---|---|
|`platform_independent`|可在 CPU 執行，包含 mocked NVIDIA／ROCm／OS routing regression|
|`nvidia`|真 NVIDIA GPU 或 native NVIDIA SDK|
|`rocm`|真 ROCm GPU；同時標 `nvidia` 表示任一 CUDA-like backend|
|`windows`／`linux`|指定 host OS；模擬 OS 的純邏輯測試不因此排除|
|`mps_real`|真正 MPS tensor／model operation|
|`model_required`|外部 checkpoints 或 proprietary model installation|

分類在 `tests/conftest.py`、marker 註冊在 `pyproject.toml`。混合 CPU／MPS 的
video tests 依 device parametrization 分類；純 CPU loading/schema／mock backend
測試仍留在 CPU suite。新增真實硬體測試時須宣告對應 marker，混合模組須逐 case
分類，不能因檔名有 AMD／MPS／Windows 就整檔跳過。

缺少 TensorRT／torch-tensorrt／tensorrt_libs 時，六個於 module scope 載入 native
SDK 的 modules 在 collection 前排除，避免 import error；不是實機 GPU 驗證通過。
`test_torch_tensorrt_export.py` 使用 mocks，仍收集執行。
`test_test_classification.py` 保護此邊界及 mixed device 分類。

```bash
cd /Users/kaho/jasna-mac/jasna
PY=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
"$PY" -m pytest --collect-only -q
"$PY" -m pytest -m 'not nvidia and not rocm and not mps_real' -q
# 只選 CPU-capable 分類（不含真 OS／外部 model cases）
"$PY" -m pytest -m platform_independent -q
```

GUI tests 需要可用的 macOS desktop／Tk；受限 sandbox 可能在 Tk 初始化 native abort，
因此不能以 abort 當測試結果或直接跳過全部 GUI tests。MPS 同樣需要 GPU 存取權限。

ROCm 的 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 保留既有 upstream policy
（`43e8bd1` 啟用 Flash／Memory Efficient Attention）。目前 base 已同步第四項 test
expectation；本議題補上每個 env default 的使用者 override regression，包含 `0`。
舊 pipeline unit tests 補齊 external CUDA memory／IPC mocks，保留原有 worker/error/
output assertions，而非將這些 mocked tests 改標 NVIDIA-only。

## 可重現 strict gate

先依 [macOS dependencies](../en/macos-development.md) 安裝目前 checkout。
Gate 使用安裝後的 console script，必須確保 editable install 指向本 checkout：

```bash
UV_CACHE_DIR=/tmp/issue13-uv-cache uv pip install --python "$PY" --no-deps --no-build-isolation -e .
WEIGHTS='/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights'
EVIDENCE=/Users/kaho/jasna-mac/verification-evidence/issue13/mps-gate
# EVIDENCE 必須是新或空目錄，避免舊輸出冒充本次結果。
env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTEST_ADDOPTS "$PY" -m scripts.verify_mps_gate \
  --weights-dir "$WEIGHTS" --output-dir "$EVIDENCE"
```

Gate 在任何缺 MPS 的 runner 上直接 FAIL。它先驗證 built／available、APPLE
自動與 explicit dispatch、真實 MPS FP32 operation，然後執行完整 `tests -m mps_real`
selection。任一 selected test skipped、failed、error、zero selection 或缺少必需核心
inference／E2E test 都 FAIL。禁止全域 MPS fallback，也禁止以 `PYTEST_ADDOPTS`
改寫已記錄的 selection。沒有 MPS 的 hosted runner 不能以 skipped jobs 宣稱成功。

`report.json` 保存 Python／Torch／torchvision／rfdetr／transformers／PyAV／NumPy／
mmengine／pytest 版本、hardware、OS、Git HEAD／diff／status、exact argv、exit codes、
JUnit cases／時間、checkpoint hashes、ffprobe JSON 及完整 decode 結果。
每條 command 的 stdout／stderr 保存為 `command-XX.log`，影片保存於 `video/`。
原 checkpoints 僅讀取並在測試前後 hash，輸出目錄不能位於 weights 目錄內。

需要四個原有 trusted checkpoints：`rfdetr-v6.pt`、`rfdetr-vr-v1.pt`、
`lada_mosaic_restoration_model_generic_v1.2.pth`、`lada_mosaic_detection_model_v4_fast.pt`。
最後兩個 detector variants 僅保留既有 loading/inference regression，不代表 CLI
的 VR／YOLO 支援已完成，也不開始 Issue #14／#18。

## 覆蓋範圍與判準

- Accelerator availability／APPLE selection／真 MPS device operation。
- CPU checkpoint loading → FP32 MPS parameters/buffers、真 detector inference 與 CPU parity。
- BasicVSR++ 兩幀／三幀 temporal inference、deform alignment、flow warp、finite／shape／dtype／device。
- PyAV software decode、RGB／PTS parity、seek／cancel／buffer lifetime、真 MPS upload。
- libx264 software encode、owned host snapshot、audio mux、PTS／color／frame count、完整 decode。
- 真 CLI mosaic fixture：正偵測、至少兩幀 restoration、輸出確實有變更、AAC 保留、取消 exit 130。
- 五次真 checkpoints threaded processing 的 memory plateau、worker cleanup 及錯誤傳播。

必要 CPU 邊界是 software decode／RGB conversion、host encode／mux、detector boxes
postprocess，以及 scan-only 非整除 area resize 的局部 CPU 路徑。模型仍真正 MPS
FP32，沒有全域或整模型 CPU fallback。NVIDIA／ROCm 實機 parity 需要各自硬體。
FP16／BF16、HDR／10-bit、VideoToolbox、GUI feature parity、VR／streaming／LTX 不在本議題。

## 本次 M2 Pro 結果

2026-10-10（香港時間），Apple M2 Pro／34359738368 bytes unified memory，
macOS 27.0.1（26A434）arm64；Python 3.12.4、Torch 2.12.0、torchvision 0.27.0、
rfdetr 1.8.3、transformers 5.1.0、PyAV 18.1.0、NumPy 2.5.3、mmengine 0.10.7、
pytest 9.1.1、FFmpeg／ffprobe 8.0.1。MPS built／available 都是 true，
automatic／explicit vendor 都是 apple，operation 在 mps:0 FP32，global fallback unset。

Strict gate：**107 passed、0 skipped、0 failed**，pytest 81.50 s，整個 gate 86.15 s。
14 個保存的影片均通過 ffprobe 及完整 FFmpeg `-xerror` decode。模型 hashes 前後相同。

- CLI 正偵測 `[1,0,1,0,0,0,0,0]`；RF-DETR outputs `(1,200,4)`、`(1,200,3)`、
  `(1,200,144,144)`，FP32／MPS／finite。
- 真 restoration `(3,3,256,256)`，mean input change > 1e-4；CLI E2E 約 3.981 s。
- 兩幀／三幀 CPU parity max abs error `5.07e-7`／`8.05e-7`。
- 五輪 allocated memory 固定 221161216 B，driver 固定 1609744384 B；
  RSS 約 2.605 → 2.609 GB，通過預先設定的 plateau ceiling。
- CLI output H.264、1920×1080、30/1、8 frames、0.266667 s，AAC 保留、PTS 遞增。
- 1080p libx264 encode：30 frames／0.565 s（53.1 fps）；decode/upload 30 frames／
  0.184 s（162.8 fps）。這是短 fixture 的單輪觀察，非長片效能承諾。

RF-DETR v6 raw output與 CPU 有小數值差異，已通過固定 tolerances、boxes/masks parity
與正偵測。既有 VR variant regression 僅保證此 fixture postprocess contract；raw logits／
masks 的裝置差異較大，不能外推為 VR feature parity。相關工作仍屬 Issue #18。

Warnings：Torch JIT／RF-DETR deprecation、DINOv2／trimmed checkpoint loading 提示、
PyAV／OpenCV 重複 libavdevice Objective-C class warnings。此次沒有造成 inference／decode
失敗；沒有藉由全域 fallback 掩蓋 warnings 或失敗。

實機 evidence：`/Users/kaho/jasna-mac/verification-evidence/issue13/mps-gate/report.json`。
最終完整 collection：2683 tests。`-m platform_independent`：2365 passed、318 deselected、
0 skipped／failed（86.92 s）。Issue 指定 `not nvidia and not rocm and not mps_real`
命令亦通過 2359 passed、22 skipped；其後細分原有可選 SDK／model cases及新增分類 tests，
CPU-capable suite 因而為零 skipped。Windows／TVAI／LTX external weights cases的 skip
不算作硬體成功。Focused policy／分類／ROCm override、mock backend regression亦通過。

另在無 GPU access 的受限 sandbox 實跑 strict gate，正確返回 exit 1／FAIL：
`Real MPS required: built=True, available=False`，沒有 skip-to-PASS。
正常 desktop 權限的真實 MPS gate 才是本次 PASS 的依據。
