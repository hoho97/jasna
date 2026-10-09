# macOS/MPS core pipeline execution（Issue #6）

## 範圍與架構審查

基於 `origin/feature/macos-mps` 的 `c145937`；直接依賴 #3 已合併。
Issue #6 與 Epic #1 均無 comments。已對照 upstream CUDA/ROCm offloader，
並閱讀既有 worker、queue、overlap、crop、blend、tracker、restoration 及 video I/O tests。

| Caller → callee | 結論／本次調整 |
|---|---|
| `Pipeline.run` → `_run_full` → `_run_pass` → `run_restoration_pass` | 共用既有 threaded pipeline；完成後透過 accelerator 顯示記憶體。 |
| `run_restoration_pass` → decode／primary／secondary／blend worker | 五個 worker entry points（含 async secondary wrapper）改用 `set_device`。 |
| decode → `VideoReader.frames` → detector → `process_frame_batch` → tracker／crop／queues | MPS batch upload、偵測、mask merge、crop clone 分段持鎖；所有 queue 等待均在鎖外。 |
| primary → `RestorationPipeline.prepare_and_run_primary` → crop prepare／BasicVSR++ | MPS 運算完成後才發布 primary result。CUDA/ROCm 不額外同步。 |
| secondary → `_run_secondary`／`build_secondary_result` → encode queue | MPS 運算與 uint8 conversion 完成後才交接。 |
| blend → `BlendBuffer.blend_frame` → frame writer／encoder | MPS 讀取、blend、host handoff 各自完成；不持鎖等待 restoration results。 |
| offloader → blend／crop buffers | MPS 停用背景 offload，不跨執行緒覆寫共享 tensor；CUDA/ROCm 保留原本 offload。 |
| LTX segment memory query | 經 device module；MPS 在查詢前由現有 capability 拒絕 LTX。未實作 LTX port。 |

`CropBuffer.split_overlap` 會共享 RawCrop 物件；`extract_crop` 已 clone GPU crop。
MPS 不執行背景 offload，避免與 primary 或 blend 同時改寫物件。
既有 `FrameQueue` 容量、metadata queue 容量及 overlap 計算不變，不新增另一套 pipeline。

## 為何需要 MPS execution lock

只在 queue 交接前呼叫 `torch.mps.synchronize()` 的第一輪真機測試，重現：

```text
-[_MTLCommandBuffer addScheduledHandler:]:807: failed assertion
'Scheduled handler provided after commit call'
Fatal Python error: Aborted
```

當時 primary padding 與 secondary uint8 conversion 在不同 Python threads 同時執行。
因此 MPS P0 使用 process-wide `RLock` 序列化運算提交與完成。
讀取 batch 的 iterator 也在鎖內取得、鎖外 yield，避免在 queue backpressure 下死鎖。
CUDA/ROCm execution context 為空操作，既有 streams／overlap 保持不變。

## 記憶體與限制

- MPS 使用 `current_allocated_memory`、`driver_allocated_memory`、`recommended_max_memory`；不呼叫 CUDA memory APIs。
- recommended working-set 是 budget，不是 free VRAM；32 GB 是整機共享容量。
- MPS driver safety threshold 為 `min(0.75 * recommended_max_memory, 0.5 * host_total) - safetynet`。
  明確指定 vram limit 只會收緊上限；預設 safetynet 為 750 MiB。
- 每 100 ms 監測一次（啟動時立即採樣）。超過 driver threshold，或 host available
  低於 `max(750 MiB, 5% * host_total)`，以具體 RuntimeError 取消並清理工作。
- 這是保守的壓力防護，不能保證阻止每一次瞬間 allocation OOM。
- P0 明確要求 `batch_size=1..2`、`max_clip_size=1..32`，超出時提供錯誤，
  不偷偷變更 clip 或 overlap。仍須符合 tracker 的 `2 * temporal_overlap < max_clip_size`。
- MPS async secondary restoration 不支援；LTX 仍不支援。
- CPU 邊界：既有 PyAV software decode／RGB staging、libx264 encode／mux；
  detector 的 boxes／tracker 在 CPU。此核心修改沒有整體模型 CPU fallback，沒有開啟
  `PYTORCH_ENABLE_MPS_FALLBACK`。

worker 或 memory monitor 出錯時 coordinator 會取消工作並 drain queues，解除阻塞的 producers。
包括 poll callback／worker 啟動途中出錯，也會 join 已啟動 threads、停止 monitor、釋放 buffers，
最後同步及清理 cache。正常完成不設置 cancel flag。

## 驗證命令

在 repository 根目錄執行；真實 MPS 命令須能存取本機 Metal（本次在 sandbox 外執行）。

```bash
PY312=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
PY313=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-313/bin/python
WEIGHTS='/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights'
EVIDENCE=/Users/kaho/jasna-mac/verification-evidence/issue6

"$PY312" --version
"$PY313" --version
"$PY312" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
"$PY313" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize
ffmpeg -version
ffprobe -version

"$PY312" -m pytest -q tests/test_pipeline_threads.py tests/test_vram_offloader.py tests/test_pipeline_overlap.py tests/test_frame_queue.py tests/test_pipeline_processing.py tests/test_pipeline_init.py tests/test_pipeline_segments.py tests/test_blend_buffer.py tests/test_crop_buffer.py tests/test_clip_tracker.py tests/test_amd_support.py tests/test_accelerator_rocm_env.py tests/test_mps_accelerator.py tests/test_mps_capabilities.py

JASNA_TEST_MODEL_WEIGHTS_DIR="$WEIGHTS" JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE/python312" "$PY312" -m pytest -q -s tests/test_mps_pipeline_runtime.py

JASNA_TEST_MODEL_WEIGHTS_DIR="$WEIGHTS" JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE/python313" "$PY313" -m pytest -q -s tests/test_pipeline_threads.py tests/test_vram_offloader.py tests/test_pipeline_overlap.py tests/test_frame_queue.py tests/test_mps_pipeline_runtime.py
```

真機 runtime tests：正常處理／重跑逐 pixel 一致、取消、detect／primary／secondary／write／memory
錯誤、poll exception、部分 thread startup failure、crop storage ownership、execution serialization、
unified memory diagnostics／停用 offload、batch／clip 安全上限。
指定 weights 後，MPS unavailable 是失敗，不會以 skip 假裝完成。

額外修正兩個既有 regression test fixtures：NVIDIA smart-render test 明確使用 `nvidia_build`；
ROCm env expected dict 補上 base 已存在的 AOTRITON default，保留完整 dict equality。
在未修改 base 的獨立 snapshot 上已重現原 ROCm test：1 failed、3 passed。

各輸出另外執行（完整十個檔案的命令／結果保存於 `video-verification.json`）：

```bash
for video in "$EVIDENCE"/python31{2,3}/pipeline-mps-*.mp4; do
  ffprobe -v error -count_frames -show_entries stream=codec_name,width,height,nb_read_frames,duration -of json "$video"
  ffmpeg -v error -i "$video" -f null -
done
```

## M2 Pro 結果（2026-10-10，香港時間）

- Apple M2 Pro，34359738368 bytes unified memory，arm64，macOS 27.0.1（26A434）。
- Python 3.12.4／3.13.13；Torch 2.12.0，Torchvision 0.27.0；MPS built／available 均 `True`。
- RF-DETR 1.8.3、Transformers 5.1.0、PyAV 18.1.0、NumPy 2.5.3、psutil 7.2.2、
  mmengine 0.10.7、OpenCV 4.14.0.94；FFmpeg／ffprobe 8.0.1。
- Regression：275 passed、2 skipped（原有硬體條件 cases）。
- Python 3.12 真機 runtime：14 passed。
- Python 3.13 focused＋真機 runtime：106 passed。

唯讀 checkpoints：

| 檔案 | SHA256 |
|---|---|
| `rfdetr-v6.pt` | `f10bedc4d105c2721e4259b8680203d51f344f73e55e85710d915619f5731b55` |
| `lada_mosaic_restoration_model_generic_v1.2.pth` | `d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee` |

測試從 `assets/test_clip1_1080p.mp4` 的第 4 秒截取真實 8 幀短片，使用原 detector threshold。
每輪偵測數為 `[1,0,1,0,0,0,0,0]`，透過既有 detection-gap=1 行為組成三幀 temporal clip。
BasicVSR++ 輸出 `[3,3,256,256]`、FP32、device=mps、finite，且確實改變輸入。
每個 Python 版本連跑五輪，輸出五個 1920×1080 H.264 MP4；各有 8 幀、
PTS 嚴格遞增、時長 0.266667 秒，PyAV 完整 decode 與 ffprobe 檢查均通過；另以系統 FFmpeg 完整 decode 全部十個輸出，均 exit 0。
原 fixture 無音訊；此測試不宣稱重新驗證音訊保留策略。

五輪後 MPS allocated 均 221161216 bytes；driver 均 1225228288 bytes。
Python 3.12 RSS 約 1.28–1.55 GB；Python 3.13 約 1.24–1.31 GB。
RSS 包含 host／FFmpeg allocator/cache，沒有宣稱它完全不增長。
五輪實際 pass 合計約 8 秒；不是完整長片效能 benchmark。

Warnings：PyAV/OpenCV 重複 AVFoundation ObjC class；Torch JIT deprecation；RF-DETR checkpoint
backbone／query inference 與 dependency deprecation。最終兩個 Python 版本測試未再出現 Metal assertion。
NVIDIA／AMD 是 regression mocks，這台機器不能執行其實體 GPU tests。
CLI 預設值／入口整合與長片 gate 屬 #12／#13，本次沒有開始實作。
