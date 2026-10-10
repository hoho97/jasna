# VideoToolbox encode／decode（Issue #15）

## 使用方式及回退政策

預設保持 PyAV software decode／libx264 encode，P0 不依賴 VideoToolbox。
Apple/MPS 可明確選用 H.264 或 HEVC 硬體編碼：

```bash
JASNA_ENCODE_BACKEND=videotoolbox JASNA_DECODE_BACKEND=pyav-hw \
python -m jasna --device mps --input input.mp4 --output output.mp4 \
  --codec hevc --encoder-settings b=8000000,g=250
```

`JASNA_ENCODE_BACKEND` 只影響 Apple：

|值|行為|設定語意|
|---|---|---|
|`software`（預設）|既有 libx264 H.264|CRF／preset，維持 #11|
|`videotoolbox`|嚴格使用硬體；session 開啟失敗即報錯|`b` 正整數 bits/second；`g` 正整數 keyframe interval|
|`auto`|嘗試硬體；codec 不存在或 session 開啟失敗時，記錄原因並改 libx264／libx265|保留相同 `b`／`g`；不轉譯成 CQ／CRF|

未指定 `b` 時使用已探測 source video bitrate；無有效 bitrate 時使用 8,000,000。
這是平均 bitrate 目標，不保證固定品質或實際檔案大小。拒絕 NVENC／AMF 的
CQ／AQ／RC／quality，也拒絕 CRF／preset，避免靜默丟棄或錯誤翻譯。
`allow_sw=0` 強制 FFmpeg VideoToolbox 使用硬體；codec 註冊後還會真正 `ctx.open()`，
成功才啟動 worker。`auto` fallback 發生在任何 frame／audio mux 前；開始編碼後的
encode／mux failure 明確傳出，不重新播放、不拼接不同 encoder 的 packets。
若 bundled FFmpeg 缺少對應 software encoder，回退仍會明確失敗。

Decode 的 `auto`／`pyav-sw` 維持 software；Apple 的 `pyav-hw` 改為嘗試
VideoToolbox，使用 `is_hw_owned=False` 取得 host frames，然後真正上傳 MPS。
裝置／codec unavailable，或首個 decoded frame 前失敗，會 log 並重開 software
reader，保留 seek／PTS／stride。交付 decoded frame 後的失敗報錯，避免重播或漏幀。
VALI 仍拒絕 Apple；沒有解除 VR／streaming／smart rendering 等 capability gates。

## Audit／架構

Base `84e810f`；已閱讀 #15、comments（無）、Epic #1 及 #10／#11／#12 body／comments。
前置 merge 都在 base：#10=`6023d0a`、#11=`c145937`、#12=`190c033`。
Epic 的部分舊 checkbox 尚未反映 merge，以 git ancestry 與 source 為準。

Caller：CLI parser／backend preflight → SessionConfig／RestorationSession → Pipeline／
DecodeDetect、BlendEncode workers → VideoReader／VideoEncoder。
Encoder 的 callee 為 RGB→YUV、CAS／LUT、PyAV frame／codec／mux、source audio／metadata
copy。沿用 blocking MPS→CPU owned snapshot，再由 encoder worker 做 CPU 色彩處理；
硬體編碼也共用這個邊界，沒有新增 CUDA stream／event／pinning 或全域 MPS sync。
NVIDIA 保持 CUDA DLPack／NVENC，AMD 保持 pinned host／AMF，其他 OS 不讀 Apple policy。

檢查 existing encoder settings/unit/mux、decode seek/backend/AMD、MPS decode/software encode、
CLI/capabilities/E2E tests、accelerator abstraction，以及 upstream/main 對應 media code；
upstream 仍只有 NVENC／AMF，沒有可直接重用的 VideoToolbox 實作。
`os_utils.check_required_executables` 已檢查 ffprobe，無須加入系統 encoder 列表 gate：
真正 encoder 能力來自 PyAV bundled libraries，而非系統 ffmpeg。

實機找出 NV12→RGB 與 yuv420p→RGB 的 swscale chroma interpolation 差異。
相同 hardware/software frame 的 YUV samples 完全一致，但直接 RGB 有差異；
僅 Apple VT NV12 frames 先重排為 yuv420p，再進既有 reformatter。
最終 H.264／HEVC、full／limited range 的完整 RGB tensors 逐像素相等。
Full-range output 的 ffprobe pix_fmt 正確為 `yuvj420p`，limited 為 `yuv420p`。

參考：[PyAV HWAccel source](https://github.com/PyAV-Org/PyAV/blob/v18.1.0/av/codec/hwaccel.py)、
[FFmpeg VideoToolbox encoder source](https://ffmpeg.org/doxygen/8.1/videotoolboxenc_8c_source.html)。

## M2 Pro verification（2026-10-10）

Apple M2 Pro／32 GiB unified memory，macOS 27.0.1（26A434）、arm64。
Python 3.12.4／3.13.13，PyTorch 2.12.0，PyAV 18.1.0；MPS built／available 均 True／True。
torchvision 0.27.0、rfdetr 1.8.3、pytest 9.1.1、NumPy 2.5.3；system FFmpeg／ffprobe 8.0.1。PyAV bundled libavcodec=62.28.102、
libavformat=62.12.102、libavutil=60.26.102。未設定 `PYTORCH_ENABLE_MPS_FALLBACK`。

在 repository root、具 host GPU access 的環境執行：

```bash
PY312=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
PY313=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-313/bin/python
EVIDENCE=/Users/kaho/jasna-mac/verification-evidence/issue15
WEIGHTS='/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights'
"$PY312" --version
"$PY312" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(),torch.backends.mps.is_available())'
"$PY313" --version
"$PY313" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(),torch.backends.mps.is_available())'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize
ffmpeg -version
ffmpeg -hide_banner -encoders

JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE/python312" JASNA_TEST_MODEL_WEIGHTS_DIR="$WEIGHTS" \
"$PY312" -m pytest -q -s tests/test_macos_videotoolbox.py \
  tests/test_mps_e2e.py::test_real_cli_detection_restoration_encode \
  --junitxml="$EVIDENCE/python312/focused.xml"
# 41 passed，無 skip；18.70 s

JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE/python313" JASNA_TEST_MODEL_WEIGHTS_DIR="$WEIGHTS" \
"$PY313" -m pytest -q -s tests/test_macos_videotoolbox.py \
  tests/test_mps_e2e.py::test_real_cli_detection_restoration_encode \
  --junitxml="$EVIDENCE/python313/verification.xml"
# 41 passed，無 skip；18.23 s

JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE/python312" JASNA_TEST_MODEL_WEIGHTS_DIR="$WEIGHTS" \
"$PY312" -m pytest -q tests/test_macos_videotoolbox.py \
  tests/test_mps_e2e.py::test_real_cli_detection_restoration_encode \
  tests/test_mps_software_encode.py tests/test_mps_video_decode.py \
  tests/test_video_decoder_seek.py tests/test_video_decoder_amd_path.py \
  tests/test_video_decoder_backends.py tests/test_video_encoder_unit.py \
  tests/test_video_encoder_mux.py tests/test_encoder_settings.py tests/test_amd_support.py \
  tests/test_mps_capabilities.py tests/test_main_cli_device.py \
  --junitxml="$EVIDENCE/python312/regression.xml"
# 367 passed，52 skipped（NVIDIA/CUDA/NVENC），34.98 s

"$PY312" -m pytest -q tests/test_main.py tests/test_main_entry.py \
  tests/test_session_factory.py tests/test_session_config.py tests/test_os_utils.py \
  tests/test_test_classification.py tests/test_verify_mps_gate.py
# 193 passed，1 skipped（Windows 專用），4.01 s

ffprobe -v error -show_streams -show_format -of json \
  "$EVIDENCE/python312/videotoolbox-hevc/cli-restored.mp4"
ffmpeg -v error -xerror -i "$EVIDENCE/python312/videotoolbox-hevc/cli-restored.mp4" -f null -
```

Focused module 40 tests，含兩個真 weights E2E；加原 software E2E 共41。
Regression 的 skips 屬 NVIDIA／CUDA／NVENC 實機測試；本機未宣稱驗證 NVIDIA／AMD hardware。
真硬體測試檢查 codec session 已開啟、`allow_sw=0`、uint8 MPS BCHW、host frame ownership、
24 frames／320×240／2 s／12 fps、BT.709 全／限幅、AAC 音訊及完整影音 decode。
另測 nonzero PTS、seek、stride、首幀取消、真10-bit source 拒絕、裝置／codec／session failure
回退、strict policy failure，以及中途 encode／decode errors 不重播。

真模型 E2E 使用原 fixture 第4秒起8幀，detector counts `[1,0,1,0,0,0,0,0]`。
RF-DETR outputs：dets `(1,200,4)`、labels `(1,200,3)`、masks `(1,200,144,144)`；
BasicVSR++ temporal output `(3,3,256,256)`；均 MPS、FP32、finite，restoration 與 input
有實際差異。軟體、硬體 H.264、硬體 HEVC 三條路徑均輸出1080p、8 frames、30 fps、
0.266667 s video／0.266 s AAC，ffprobe／完整 decode 通過。

唯讀 weights（未修改原檔）：

|檔案|SHA256|
|---|---|
|rfdetr-v6.pt|f10bedc4d105c2721e4259b8680203d51f344f73e55e85710d915619f5731b55|
|lada_mosaic_restoration_model_generic_v1.2.pth|d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee|

Performance observation：8-frame E2E，Python3.12 VT H.264約4.13 s、HEVC約2.86 s；
Python3.13約3.94／2.73 s。包含模型載入／初次暖機／mux，不是單獨 encoder benchmark，
不推論硬體編碼相對 software 的速度或長片品質。既有 Torch JIT／rfdetr deprecated warnings
及 checkpoint patch-size／num_queries 提示仍存在，不影響本次 contracts。

## 驗收／限制

- [x] H.264／HEVC 實機硬體 encode，ffprobe／完整 decode 通過。
- [x] 明確 strict／auto 回退政策，log 原因，保留 bitrate 語意，不丟品質選項。
- [x] decode unavailable 不阻擋預設 P0；PTS／色彩／8-bit 正確，拒絕10-bit／HDR。
- [x] 不使用 NVENC／AMF quality options；保留 capability／routing tests。

Result：PASS。沒有 Issue #15 blocker。只承諾 even dimensions 的8-bit SDR；
10-bit／HDR、odd sizes、AV1、zero-copy Metal interop、VR／streaming／smart splice 未開放。
CPU 邊界為 host RGB snapshots、YUV／RGB conversion、CAS／LUT／mux；auto encode fallback
為 libx264／libx265，decode fallback 為 FFmpeg software。模型保持 MPS FP32。
沒有修改 batch／clip 限制，沒有開始 #30 或其他後續 Issue。
