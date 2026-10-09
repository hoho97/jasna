# Issue #12：M2 Pro 長影片實測（原專案 batch／clip）

Result：**PASS（處理、模型 contract、完整影音解碼及回歸測試）**。
Windows 版主觀畫質是否持平，待使用者查看成品；本次沒有 Windows 對照片或畫質評分。

## 設定及極小相容性調整

2026-10-10（香港時間），branch `macos-mps/issue-12`，Draft PR #29。
使用原專案 batch **4**、max clip **90**、temporal overlap **8**，沒有在執行中降低設定。
RF-DETR v6 → BasicVSR++，MPS FP32、compile=false、secondary=none、VR off；
software libx264 H.264，CRF23／preset medium。Windows／CUDA 原 defaults 不變。

先前 MPS CLI／pipeline 只允許 explicit batch<=2、clip<=32；本次將兩處 gate
最小幅度擴至 batch<=4、clip<=90，並新增 CLI／真實 MPS workers regression。
保守 MPS 預設仍為 batch1／clip16／overlap2；memory budget、同步／取消、模型與 encoder
路徑未改。沒有實作其他 issues，沒有整個模型 CPU fallback，也沒有設定
`PYTORCH_ENABLE_MPS_FALLBACK=1`。既有 CPU 邊界仍為軟體解碼、RGB conversion、
小量 boxes postprocess、MPS→host 軟體編碼與 mux。

## 環境及檔案

- Apple M2 Pro、12 logical CPU、32 GiB unified memory（34359738368 bytes）。
- macOS 27.0.1／26A434、arm64；Python 3.12.4、PyTorch 2.12.0。
- 真實 `torch.backends.mps.is_built()`／`is_available()` 均為 **True**。
- PyAV 18.1.0、FFmpeg／ffprobe 8.0.1；其他依賴版本見 [CLI 驗證](macos-mps-cli.md)。
- 來源：`/Users/kaho/jasna-mac/test_clip.mp4`，未修改。
- 成品：`/Users/kaho/jasna-mac/test_clip_mps_batch4_clip90.mp4`。
- 本機證據：`/Users/kaho/jasna-mac/verification-evidence/issue12-long/`。
- 原始 weights 以唯讀方式使用：`/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights`。
  實際載入 `rfdetr-v6.pt` 與 `lada_mosaic_restoration_model_generic_v1.2.pth`；未覆寫或修改。
  本輪重新計算 SHA256，與 CLI 短片紀錄及 THIRD_PARTY_MODELS 一致（見 weights-sha256.json）。

|資訊|來源|成品|
|---|---|---|
|檔案大小|810175976 bytes（772.64 MiB）|495621737 bytes（472.66 MiB）|
|容器時長|998.325 s（16:38.325）|998.338 s（16:38.338）|
|視訊時長|998.325 s|998.330667 s|
|視訊|H.264 Main|H.264 High（libx264）|
|尺寸／像素格式|1920×1080／yuv420p|1920×1080／yuv420p|
|色彩|BT.709／limited（tv）|BT.709／limited（tv）|
|nominal frame rate|30000/1001|30000/1001|
|average frame rate|8584000/287001 ≈ 29.9093|11197125/374374 ≈ 29.9089|
|實際完整解碼幀數|29859|29859|
|視訊 bitrate|6068683 bit/s|3579755 bit/s|
|音訊|AAC／48 kHz／stereo／383228 bit/s|AAC LC／48 kHz／stereo／383228 bit/s|
|音訊封包數|46705|46705|

來源容器 metadata 的 `nb_frames=30044` 並不等於實際可解碼幀數。PyAV 完整解碼
來源與成品均為 **29859**，與 pipeline 實際處理數一致，不能用 metadata 差額當成
輸出遺失幀。兩者首 PTS=0、末 PTS=998.2973 s，PTS 均嚴格遞增。
容器時長差 13 ms；音訊封包 payload 數量、總長度與 SHA256 完全相同，沒有重編音訊：
`04b2f6928f75647807e0d8f63934361ea7b14dd8a809d473e845e6091623c17e`
（46705 packets／47825910 bytes）。

## 時間及資源

開始 04:02:15，轉碼結束 05:54:40；**6745.63 s（1:52:25.63）**，包含模型載入及
pipeline cleanup，不含後續完整解碼驗證（另 142.88 s）。實際 throughput **4.426 fps**，
約 **0.148× realtime**，即影片時長的 **6.76 倍**。

|指標|平均|取樣峰值|
|---|---:|---:|
|process RSS|1.220 GiB|3.505 GiB|
|MPS live allocated|0.614 GiB|1.890 GiB|
|MPS driver allocated（含 allocator／driver 保留）|10.206 GiB|13.277 GiB|
|全機 host used（psutil）|14.857 GiB|18.205 GiB|
|全機 swap used|2.270 GiB|6.054 GiB|
|process CPU（100%=一個核心）|97.48%（樣本均值）|309.7%|
|全機 CPU（100%=所有核心）|17.47%|51.0%|
|全機 GPU Device Utilization|95.27%|98%|
|全機 GPU Renderer／Tiler Utilization|6.86%／6.86%|94%／94%|

CPU 累計 user 4142.88 s、system 2475.56 s；除以轉碼 wall time，process CPU
整段平均 **98.11%**（約 0.981 核），比樣本均值更適合表示整段 CPU 消耗。
全機 available RAM 最低 **4.293 GiB**。沒有 OOM 或 safety-budget failure；
MPS driver 取樣峰值低於既有 15634 MiB safety threshold。driver 保留量由約 4 GiB
增至約 13 GiB，swap 由 0 增至約 6 GiB；此設定並非低記憶體負載，不能用 RSS 單獨
代表整個 unified-memory 成本，也不能由一次成功推論任意長片的 memory plateau。

測量限制：

- 1086 個樣本，監控每輪等待 5 s；包含採樣工作／排程後實際間距平均 6.213 s、
  最長 7.856 s。表中是樣本算術均值及取樣峰值，不能保證捕捉瞬時最高值。
- RSS 與 MPS driver 位於 unified memory，可能重疊，**不可相加**。
- 全機 host／swap／CPU／GPU 包含其他背景程式，不能全歸因於 Jasna。
- GPU 來自 IORegistry `AGXAccelerator/PerformanceStatistics`，是整機利用率，
  不是 Jasna 專屬 GPU 時間或記憶體；renderer／tiler 與 device 讀值分別列出。
- `powermetrics` 因 `sudo -n` 需要密碼不可用，未取得 per-process GPU／功耗。
- 每個 restored clip 有 finite assertion，監控也有開銷；不是無儀器的效能 benchmark。
- `caffeinate -i -w PID` 僅於處理程序存活時防止 idle system sleep，未改系統設定。

## 實際模型與輸出驗證

- RF-DETR 真實呼叫 **7465** 次，處理 **29859** frames；**20431** positive frames，
  **21268** detections。不是零偵測 passthrough；沒有替換模型輸出。
- BasicVSR++ 真實 restoration **496** 次，總輸入 **24560** clip frames（含重疊／不同 track，
  不能當成 unique frame count），每次回傳 `(T,3,256,256)`、MPS、FP32、finite。
  實際 temporal length **2..90**，已真正跑到 clip90。
- 來源／成品 PyAV 完整 video decode、相同 frame count、嚴格遞增 PTS：PASS。
- FFmpeg `-xerror` 完整 audio＋video decode：exit0，error log 為空。
- AAC 封包 payload hash 比對：PASS。
- 無處理失敗，無中途 retry，無降 batch／clip，無模型 CPU fallback。

## 實際命令與測試結果

於 `/Users/kaho/jasna-mac/jasna` 執行；真 MPS 及 GitHub 操作使用 host GPU／network access。
觀測 runner 保留於本機 evidence，呼叫原 production `jasna.main.main()`，
只包裝 detector／restorer 做 contract assertions 及 counters，未 mock inference。

```bash
PY=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
E=/Users/kaho/jasna-mac/verification-evidence/issue12-long
"$PY" -u "$E/run_long.py" > "$E/run.log" 2>&1
# exit0；processing-summary.json=PROCESSING_PASS；validation.json=PASS
# runner 實際傳入的 CLI arguments：
# jasna --device mps --batch-size 4 --max-clip-size 90 --temporal-overlap 8 \
#   --input /Users/kaho/jasna-mac/test_clip.mp4 \
#   --output /Users/kaho/jasna-mac/test_clip_mps_batch4_clip90.mp4 --no-progress --log-level info

"$PY" --version
"$PY" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize
# Python3.12.4 / Torch2.12.0 / True True / macOS27.0.1 / arm64 / M2Pro / 34359738368

ffprobe -v error -show_format -show_streams -of json /Users/kaho/jasna-mac/test_clip.mp4
ffprobe -v error -show_format -show_streams -of json /Users/kaho/jasna-mac/test_clip_mps_batch4_clip90.mp4
ffmpeg -v error -xerror -i /Users/kaho/jasna-mac/test_clip_mps_batch4_clip90.mp4 -f null -
# probe及完整decode均exit0；PyAV frame/PTS及音訊hash由runner／觀測程式記錄

ioreg -r -c AGXAccelerator -a
# monitor使用plist PerformanceStatistics；psutil CPU/RAM與torch.mps memory API見run_long.py

"$PY" -m pytest -q tests/test_main_cli_device.py tests/test_main_entry.py tests/test_session_factory.py
# 49 passed；focused-final.log

"$PY" -m pytest -q tests/test_main_cli_device.py tests/test_main_entry.py tests/test_session_factory.py \
  tests/test_main.py tests/test_main_validation.py tests/test_mps_capabilities.py \
  tests/test_amd_support.py tests/test_encoder_settings.py tests/test_engine_compiler.py \
  tests/test_detection_registry.py tests/test_model_weights_dir.py tests/test_post_export_action.py \
  tests/test_accelerator_rocm_env.py tests/test_session_config.py
# 323 passed / 1 skipped（CUDA GPU test）；regression.log

"$PY" -m pytest -q tests/test_mps_pipeline_runtime.py tests/test_mps_video_decode.py tests/test_mps_software_encode.py
# 77 passed / 1 skipped（既有五次真weights plateau opt-in case）；mps-regression.log
# 新增的4/90/8 worker case實際在MPS通過；長片本身使用真weights
```

前一輪短片 checkpoints E2E **2 passed** 及 CUDA pipeline tests 在 base 可重現的
14 failures／18 passes，仍以 [CLI 驗證紀錄](macos-mps-cli.md) 為準；本次未聲稱全 suite 全綠。

## Warnings／風險及 acceptance checklist

既有 PyAV／OpenCV 重複 libavdevice Objective-C class、torch.jit／rfdetr deprecation、
DINOv2/checkpoint args 提示仍出現；mux 有 `track 1: codec frame size is not set` warning。
完整影音 decode 及音訊 hash 通過，未為消除提示而修改外部依賴。

- [x] 完整來源以原 batch4／clip90／overlap8 處理。
- [x] 真實 M2 Pro／MPS available，detector MPS FP32、restorer MPS FP32／finite、nonzero restoration。
- [x] 成品完整解碼、frame count／PTS 檢查、音訊 payload 保留。
- [x] 時間、RAM／CPU／GPU、swap、影片資訊及測量限制已記錄。
- [x] 新增與 relevant regression tests 通過，NVIDIA／AMD defaults 未更動。
- [ ] 使用者觀看成品，判斷是否與 Windows 主觀畫質持平。

沒有已知阻擋本次轉碼的 blocker。仍未驗證 NVIDIA／AMD 實機、任意長度影片的
memory plateau、Windows 畫質 parity 或其他編碼／模型。成品 H.264 CRF23 與 Windows
可能使用的硬體編碼設定不同，僅 batch／clip／overlap 對齊原專案預設。
