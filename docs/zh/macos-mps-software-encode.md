# Software H.264 encode 與 MPS→host handoff（Issue #11）

## Summary

`VideoEncoder` 現在可接收真正 MPS uint8 CHW RGB frame，輸出 PyAV libx264
8-bit H.264 MP4。預設 CRF 23／preset medium；software backend 與 hardware
frame path 分開，NVIDIA NVENC 與 AMD AMF 的設定、同步、mux 行為保留。

## Why

基準 `origin/feature/macos-mps`：`6023d0a`。完整讀取 Issue #11、Epic #1
（兩者均無 comments），並確認 dependencies #3（`560b7c3`）與 #7
（`70316c3`）均為目前 base 的 ancestor。#10 software decode 亦已合併。
`origin` 是 `hoho97/jasna`；upstream `Kruk2/jasna` 僅唯讀參考。

原 encoder 明確拒絕 Apple／CPU；resolver 的非 AMD 分支選 NVENC，
`__enter__` 選 `pix_fmt=cuda`。AMD host staging 仍依賴 ROCm stream／pinned memory，
不能直接給 MPS 使用。系統 FFmpeg 與 PyAV bundled codecs 是不同的 libraries；
本次實際查詢 `av.Codec('libx264', 'w')` 並完成 encode，不依靠系統 encoder list 推斷。

## Implementation

- `EncoderSpec.backend` 明確區分 software／hardware。Apple／CPU 使用獨立
  libx264 spec；software options 僅接受 `crf,preset,g,bf,maxrate,bufsize`。
  不傳 NVENC CQ／AQ／lookahead、AMF quality options 或隱含 source bitrate cap。
- `encode()` 在回傳前以 `to('cpu', non_blocking=False, copy=True,
  memory_format=torch.contiguous_format)` 取得獨立 host RGB snapshot。
  支援 strided RGB input，caller 可立即重用原始 decoder／blend buffer。
- 重用 CPU `RgbToYuvConverter` eager fallback → NV12；PyAV 複製到獨立 frame
  planes 並轉為 libx264 的 yuv420p。worker 重用 packed／scratch buffers，
  encoder 的延遲 B-frame 不會看到下一張 frame 的資料。
- software 路徑不使用 pinned memory、HWAccel、CudaContext、CUDA streams／events。
  選用 LUT／CAS 在同一個 host encoder 邊界執行；不改 model／restorer 的 device。
- 保留既有 PTS reorder／time_base／color tags／SAR／metadata／音訊 mux。
  相容 AAC packet copy；不相容 WMA 轉 AAC，沿用現有 resampler／flush。
- worker device setup 失敗也繼續 drain queue，close 不會卡在 task_done；
  `__enter__` 失敗會關閉已開啟的 containers。worker／codec 開啟錯誤向 caller 傳遞。
- CLI settings resolver 僅調整 encoder 設定介面：Apple 可驗證 CRF／preset，
  明確拒絕 `--cq`；沒有接通 #6/#12 的完整 pipeline 或更改預設 CLI codec。

Caller/callee review：`Pipeline._video_encoder`／`_OfflineFrameWriter.write` →
`VideoEncoder.encode` → reorder buffer → bounded worker queue → RGB/YUV converter →
PyAV encode → `_mux_video`／source stream copy/transcode → flush／close。
亦檢查 SessionConfig、smart-fragment factory、StreamingEncoder、encoder settings、
accelerator capabilities／stream helpers、CAS／LUT／YUV scratch，及 NVIDIA／AMD tests。
upstream encoder 的相關實作與修改前相同，仍只提供 NVENC／AMF host path。
SessionConfig 的 codec literal 已含 h264，不需變更；streaming 與 smart rendering
延續現有 MPS capability gate，不在此 Issue 提前實作。

實測 debug：初版讓 worker 在 MPS 做色彩轉換，producer 使用全域 MPS synchronize
保護 snapshot，觸發 Metal `commit command buffer with uncommitted encoder` native abort。
採取上述 blocking host snapshot／局部 CPU conversion 後，雙 encoder worker／重用
frame 的四次並行輸出，以及 Python 3.12/3.13 真實 MPS 測試全部通過。
沒有全域 `PYTORCH_ENABLE_MPS_FALLBACK`、整個 model CPU fallback 或 Metal custom kernel。

## Tests

在 repository root 執行，GPU tests 使用能存取本機 MPS 的環境：

```bash
PY312=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
PY313=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-313/bin/python
EVIDENCE=/Users/kaho/jasna-mac/verification-evidence/issue11

JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE" "$PY312" -m pytest -q -s tests/test_mps_software_encode.py tests/test_video_encoder_unit.py tests/test_video_encoder_mux.py tests/test_encoder_settings.py tests/test_amd_support.py tests/test_mps_video_decode.py tests/test_mps_accelerator.py tests/test_mps_capabilities.py tests/test_main_cli_device.py
# 282 passed, 45 skipped。

JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE/python313" "$PY313" -m pytest -q -s tests/test_mps_software_encode.py tests/test_video_encoder_unit.py tests/test_video_encoder_mux.py tests/test_encoder_settings.py tests/test_amd_support.py tests/test_mps_video_decode.py tests/test_mps_accelerator.py tests/test_mps_capabilities.py tests/test_main_cli_device.py
# 282 passed, 45 skipped。

JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE" "$PY312" -m pytest -q -s tests/test_mps_software_encode.py
JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE/python313" "$PY313" -m pytest -q -s tests/test_mps_software_encode.py
# 最終加入 strided input 後，兩個 Python 均 29 passed，沒有 CPU/MPS skip。

git diff --check
# PASS。
```

45 skips 是 44 個既有 CUDA hardware mux cases 與 1 個 AMD GPU case；
NVIDIA／AMD settings、frame-path、worker／mux 邊界以現有 regression tests 驗證，
本機沒有這兩種 GPU，不能宣稱其實機驗證。Full suite 未執行。

新增測試包含真實 MPS frames／software decode→encode、ffprobe／FFmpeg 完整 decode、
AAC copy／WMA→AAC、原始 PTS（包括 Matroska 毫秒量化）、VFR、full／limited luma、
strided／立即重用的 frame、B-frame flush、並行 encoder、1080p、LUT／CAS、empty job、
body cancellation、worker error／device setup error、實際 libx264 option error、
container open failure cleanup、硬體設定拒絕與 SDR／even dimension guards。

## M2 Pro verification

2026-10-10：Apple M2 Pro、34359738368 bytes（32 GiB）unified memory、
macOS 27.0.1／26A434、arm64。Python 3.12.4／3.13.13，Torch 2.12.0，
兩個環境 `mps.is_built()/is_available()` 都為 True/True。
PyAV 18.1.0，NumPy 2.5.3；PyAV bundled libavcodec 62.28.102、libavformat
62.12.102、libswscale 9.5.102。系統 FFmpeg／ffprobe 8.0.1。

```bash
"$PY312" --version
"$PY312" -c 'import torch,av,numpy,os; print(torch.__version__); print(torch.backends.mps.is_built(),torch.backends.mps.is_available()); print(av.__version__,numpy.__version__); print(av.library_versions); print("PYTORCH_ENABLE_MPS_FALLBACK",os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))'
"$PY313" --version
"$PY313" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(),torch.backends.mps.is_available())'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize
ffmpeg -version
ffprobe -version

ffprobe -v error -count_frames -show_streams -show_format -of json "$EVIDENCE/encode-mps-aac.mp4"
ffprobe -v error -count_frames -show_streams -show_format -of json "$EVIDENCE/encode-mps-wmav2.mp4"
ffprobe -v error -count_frames -show_streams -show_format -of json "$EVIDENCE/encode-mps-1080p.mp4"
ffmpeg -v error -xerror -i "$EVIDENCE/encode-mps-aac.mp4" -f null -
ffmpeg -v error -xerror -i "$EVIDENCE/encode-mps-wmav2.mp4" -f null -
ffmpeg -v error -xerror -i "$EVIDENCE/encode-mps-1080p.mp4" -f null -
```

上述 probe／完整 decode 由測試實際執行，JSON 保存於影片旁。
96×64 輸出：H.264 yuv420p、24 frames、video 約 2 秒；AAC copy／WMA→AAC 音訊均存在。
WMA source 為 Matroska，保留其毫秒量化 PTS，video duration 2.000313 秒，
AAC duration 約 2.043 秒（codec padding）；A/V 起點偏差 <50ms。
1080p：H.264 1920×1080、30 frames、video 1 秒，可完整 decode；逐幀 luma 符合輸入。
此 Issue 不需 model weights，沒有讀寫或修改原始唯讀 weights。

## Expected / observed result

- [x] 真實 MPS frame → H.264 MP4；flush／close／worker errors 正確。
- [x] ffprobe codec／dimensions／duration／frame count 正確；PyAV 與 FFmpeg 完整 decode。
- [x] AAC 保留、不相容音訊明確轉 AAC；實際 luma range／原始 PTS 正確。
- [x] software 路徑不建立 HWAccel／CudaContext；NVIDIA／AMD structural regression 通過。

短 1080p 灰階序列的 handoff＋CPU conversion＋libx264 30 frames：
Python 3.12 約 0.378 秒（79.3 fps）、Python 3.13 約 0.276 秒（108.8 fps）。
這是單次、短片、低複雜度合成畫面觀察，包含 encoder flush，並非真實影片品質或
完整 detection/restoration pipeline benchmark；時序／負載會影響結果。

## Risks

CPU RGB snapshot 增加 host bandwidth／queue memory；長片、4K／8K、持續 memory peak
及複雜內容速度未經此次驗證。MPS→host 為刻意 blocking boundary。
libx264 缺失時明確報錯；不會靜默選用 NVENC／AMF 或不同 codec。

既有 `torch.jit.interface` deprecation，以及 PyAV/OpenCV 的 duplicate
libavdevice Objective-C class warnings 可見；最終測試沒有 native abort。
系統 FFmpeg 不含 libvorbis，測試因此用真正 WMA→AAC 驗證不相容音訊轉碼，沒有跳過。

沒有阻止 Issue #11 交付的 blocker。CLI 完整端到端仍屬 #12，core worker／memory
整合屬 #6；此驗證不代表整個 restoration pipeline 已接通。

## Unsupported / fallback behavior

只支援 software 8-bit SDR H.264／even dimensions；HEVC、AV1、10-bit、PQ／HLG、
smart fragments 明確拒絕。沒有隱含 HDR tone mapping。Streaming／VR／smart rendering
不在此次支援範圍；VideoToolbox 留待 #15。

局部 CPU 路徑是 RGB host snapshot、RGB→YUV、選用 LUT／CAS、libx264 encode、
audio conversion／mux。MPS model execution、NVIDIA NVENC、AMD AMF 不改。

## Issue

Closes #11
