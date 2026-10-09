# macOS/MPS CLI 最短端到端路徑（Issue #12）

沿用 `jasna.main` → `SessionConfig` → `build_restoration_session`／`build_pipeline`
→ threaded decode/detect、primary restore、blend/encode。沒有新增 pipeline，
沒有把 MPS 當 CUDA，也沒有整個模型的 CPU fallback。

## 預設值及啟動檢查

CLI 在解析完 `--device` 後才填入未指定的 defaults。Apple host 自動選 `mps`；
也可明確使用 `--device mps`。明確選 `cuda:0` 仍使用 NVIDIA／AMD defaults。
`--help`／`--version` 在載入 Torch／模型或 GUI 之前就返回。

|設定|Apple/MPS|NVIDIA／AMD（維持原值）|
|---|---|---|
|device|mps|cuda:0|
|batch size|1|4|
|fp16|false|true|
|compile BasicVSR++|false|true|
|max clip size|16|90|
|temporal overlap|2|8|
|VR mode|off|auto|
|codec|h264（libx264）|hevc|
|detector|RF-DETR v6|RF-DETR v6|
|secondary|none|none|

MPS 的 explicit batch 接受 1..4、clip 1..90（包含原專案的 batch 4／clip 90），並保留 SessionConfig 的
`2 * temporal_overlap < max_clip_size` 等驗證；不會悄悄 clamp 使用者的值。
FP16、TensorRT compile、HEVC／AV1、CQ、其他 detector、VR（含 auto）、LTX、
secondary、streaming、segments、benchmark、supporter options 於啟動時拒絕。
仍僅支援單一 8-bit SDR 影片；image／folder 及 HDR／10-bit 仍由既有 guards 拒絕。
software quality 使用 CRF／preset，而不是硬體 CQ。

SIGINT 在執行 pipeline 時設定既有 cancel event，讓 worker 收尾，返回 130；
正常返回 0，處理失敗返回 1，argparse／不支援選項返回 2。取消不執行成功後指令，
finally 還原原 signal handler 並關閉 session。取消可能留下不完整輸出，不代表成功。

真實首幀取消測試發現 PyAV AUTO frame threading 在 macOS 提早關閉 decoder 時
可能卡死：faulthandler 顯示 decode worker 在 native teardown、其他 workers 等待，
main 卡在 join。單純改 signal handling 或先關 iterator 仍失敗；Apple software
codec 改用 `thread_count=1` 後同一 SIGINT 測試成功。這是極小的 #10 compatibility
adjustment；CPU／NVIDIA／AMD decoder threading 維持不變，沒有加入 timeout 強殺或
跳過 cleanup。CPU decode throughput 可能下降，後續效能調整需保留此取消 regression。

## 可複製命令

依 [macOS dependencies 安裝說明](../en/macos-development.md) 安裝 `.[macos]`。
在已安裝 dependencies 的驗證環境，本次用以下命令安裝目前 checkout：

```bash
cd /Users/kaho/jasna-mac/jasna
PY=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
CLI=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/jasna
WEIGHTS='/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights'
EVIDENCE=/Users/kaho/jasna-mac/verification-evidence/issue12
UV_CACHE_DIR=/tmp/issue12-uv-cache uv pip install --python "$PY" --no-deps --no-build-isolation -e .
"$CLI" --help
"$CLI" --version
env -u JASNA_MAIN_PID "$PY" -m jasna --help

# 任意合適的單一 8-bit SDR input；原始 weights 唯讀。
JASNA_MODEL_WEIGHTS_DIR="$WEIGHTS" "$CLI" --device mps \
  --input input.mp4 --output restored.mp4 --no-progress --log-level info
# 選配 software quality：--encoder-settings crf=18,preset=fast
```

不要設定 `PYTORCH_ENABLE_MPS_FALLBACK=1` 來掩蓋失敗。software decode／RGB conversion、
host encode／mux 及 detector 小量 boxes postprocess 是既有明確 CPU 邊界；
RF-DETR 與 BasicVSR++ 真正於 MPS FP32 執行。本路徑沒有呼叫 scan-only area-resize fallback。

## M2 Pro 實測（2026-10-10，香港時間）

Apple M2 Pro／34359738368 bytes unified memory；macOS 27.0.1（26A434），arm64。
Python 3.12.4；Torch 2.12.0、Torchvision 0.27.0、rfdetr 1.8.3、transformers 5.1.0、
PyAV 18.1.0、mmengine 0.10.7、NumPy 2.5.3；FFmpeg／ffprobe 8.0.1。
GPU commands 必須有實際 GPU access；受限 sandbox 的 available=false 不代表這台 Mac 沒有 MPS。

```bash
"$PY" --version
"$PY" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize
"$PY" -c 'import torch; print(torch.ones(4, device="mps").mul(2))'
ffmpeg -version
ffprobe -version
```

MPS built=true／available=true；tensor 為 `mps:0`，值 `[2,2,2,2]`。
Issue 指定的三個 focused files 共 **48 passed**；下列擴大 focused／regression 為
**322 passed、1 skipped**（跳過需要 CUDA 的 YUV GPU test）；真實 MPS worker／decode／encode
為 **76 passed、1 skipped**（本輪未 opt-in 既有五次 checkpoint memory-plateau test）；
新增實機 E2E 為 **2 passed**，有實際載入 checkpoints，沒有 skipped inference。

```bash
"$PY" -m pytest -q tests/test_main_cli_device.py tests/test_main_entry.py tests/test_session_factory.py \
  tests/test_main.py tests/test_main_validation.py tests/test_mps_capabilities.py \
  tests/test_amd_support.py tests/test_encoder_settings.py tests/test_engine_compiler.py \
  tests/test_detection_registry.py tests/test_model_weights_dir.py tests/test_post_export_action.py \
  tests/test_accelerator_rocm_env.py tests/test_session_config.py

"$PY" -m pytest -q tests/test_mps_pipeline_runtime.py tests/test_mps_video_decode.py tests/test_mps_software_encode.py

JASNA_TEST_MODEL_WEIGHTS_DIR="$WEIGHTS" JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE" \
  "$PY" -m pytest -q -s tests/test_mps_e2e.py
```

E2E 使用 repository 既有 `assets/test_clip1_1080p.mp4` 第 4 秒起 8 幀（不是網路新下載），
以 FFmpeg libx264 CRF12 加上合成 440 Hz AAC 音訊。所有 inference 呼叫原 production
models；wrappers 只觀察與 assert contracts，不替換 inference 或 detections：

- 8 次 RF-DETR inference，detections `[1,0,1,0,0,0,0,0]`，threshold 仍為預設 0.35。
- RF-DETR：dets `(1,200,4)`、labels `(1,200,3)`、masks `(1,200,144,144)`；FP32、MPS、finite。
- BasicVSR++：真正 temporal output `(3,3,256,256)`；FP32、MPS、finite，與 normalized input 的平均絕對差 > 1e-4。
- 預設 tracking／scene detection／gap／min-duration，沒有以零偵測 passthrough 充當成功。
- output H.264、1920×1080、30/1 fps、8 frames、0.266667 s，PTS 嚴格遞增。
- AAC 0.266 s、13 decoded audio frames；PyAV 全 video decode 及 FFmpeg `-xerror` 全 decode 通過。
- installed console CLI 實跑成功；subprocess 自動選 MPS，真實 SIGINT=130、invalid media=1、unsupported FP16=2，失敗／取消不執行成功後指令。
- session detector／restorer 關閉、workers 不殘留，並禁止 CUDA calls。沒有全域 MPS fallback。

```bash
# E2E 建立 cli-source.mp4 後，實際安裝後的命令也已執行：
JASNA_MODEL_WEIGHTS_DIR="$WEIGHTS" "$CLI" --device mps \
  --input "$EVIDENCE/cli-source.mp4" --output "$EVIDENCE/installed-cli-restored.mp4" \
  --no-progress --log-level info
ffprobe -v error -count_frames -show_entries \
  stream=codec_name,codec_type,width,height,nb_read_frames,duration,avg_frame_rate \
  -of json "$EVIDENCE/cli-restored.mp4"
ffmpeg -v error -xerror -i "$EVIDENCE/cli-restored.mp4" -f null -
shasum -a 256 "$WEIGHTS/rfdetr-v6.pt" "$WEIGHTS/lada_mosaic_restoration_model_generic_v1.2.pth"
```

原始 checkpoint hashes 與 `assets/THIRD_PARTY_MODELS.md` 一致：

- rfdetr-v6.pt：`f10bedc4d105c2721e4259b8680203d51f344f73e55e85710d915619f5731b55`
- lada_mosaic_restoration_model_generic_v1.2.pth：`d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee`

## 範圍與 regression 限制

前置 #4、#5、#6、#8、#9、#10、#11 已合併於 base `a16c7ca`；upstream 為相同原始
`d366827`，只作對照。本變更只補 CLI integration、必要解碼相容性及其 tests；
未開始 #13 的全 repository test 分類／CI gate，也沒有 GUI／VR／VideoToolbox 工作。

NVIDIA／AMD CLI defaults、device context、CQ／AMF／NVENC、compiler／session mock contracts
有 regression 保護；此機無 NVIDIA／AMD GPU，不能聲稱實機 vendor parity。
另跑舊 `test_pipeline_run.py`／`test_pipeline_run_sync.py`：14 failed／18 passed，
同樣 14 個缺 CUDA mock 的失敗在未修改 base `a16c7ca` 重現，主要為 `VramOffloader`
查詢 CUDA device properties。沒有為通過而弱化 assertions；完整 suite 未宣稱全綠。

有既有 torch.jit／rfdetr deprecation、DINOv2／trimmed checkpoint 提示，及 PyAV／OpenCV
重複 libavdevice Objective-C class warnings。已驗證本次 decode／inference／encode；
長影片、其他 dependency versions、FP16／BF16、HDR／10-bit 與硬體 encode 未在本次聲稱支援。
installed CLI 短 fixture 觀察：decode/detect worker 約 2.0 s、primary restore 約 0.3 s，
MPS driver memory sample peak 1764 MiB、offloads=0、結束 RSS 約 1243 MiB。
這些時間／memory 只作觀察，不能外推為長片 benchmark 或品質保證。

## 原專案 batch／clip 長影片追測

2026-10-10 使用 batch4／clip90／overlap8 完整處理 16:38 長片，真實 MPS FP32，
轉碼 1:52:25.63，實際 29859 frames，完整影音 decode 通過。保守 MPS 預設仍為1／16／2。
詳細資源、swap、exact commands、tests、warnings 與畫質限制見
[長影片實測紀錄](macos-mps-long-video-verification.md)。
