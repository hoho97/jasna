# Apple/MPS throughput profiling（Issue #31）

## 範圍與架構審查

基線為 `origin/feature/macos-mps` 的 `c0db2b0`；#15 已由 PR #34 合併。
#6、#8、#9、#10、#11、#12、#13 的前置實作均已在 base；Epic 的舊 checkbox
不作為 Git 合併狀態的替代。本次沒有實作 #30 的 batch/clip 上限調整。

呼叫關係：`main → RestorationSession → pipeline → run_restoration_pass`。
DecodeDetect 透過 `VideoReader`、`process_frame_batch` 呼叫 RF-DETR preprocess／
Torch runner／postprocess，再交給 tracker／crop buffers。PrimaryRestore 透過
`RestorationPipeline.prepare_and_run_primary → BasicvsrppMosaicRestorer.raw_process`
呼叫 BasicVSR++ 的 SPyNet、四個 propagation branches、deform alignment、reconstruction。
SecondaryRestore 執行原有 crop conversion；BlendEncode 重新 decode source、blend，
再交給 `VideoEncoder.encode` 的 owned CPU snapshot 與 encoder worker。

MPS execution context 的 process-wide RLock 保護 submission 與 completion，queue waits
在鎖外。NVIDIA／ROCm 保留 streams、TensorRT／Torch、NVENC／AMF；profiling 只由獨立
script 啟動，普通 application 不會 import 或安裝 observation patches。

唯一 production optimization：Apple/MPS 使用 rfdetr 1.8.3 已有的
`optimize_for_inference(compile=False, dtype=torch.float32, inplace=True)`。
在 CPU 完成 checkpoint 載入及 export，再移到 MPS；省去一般 forward 計算、但 Jasna
不消費的 auxiliary／encoder segmentation outputs。三個公開 tensors 的名稱、shape、
FP32 與 device contract 保持一致。不新增 compile、模型替換、quality reduction、CPU
model fallback 或第三方套件全域 monkey patch。

`JASNA_MPS_RFDETR_EAGER=1` 明確選回舊 MPS forward，用於診斷與 before/control；
啟動 log 會列出選擇。NVIDIA／AMD 不受此設定影響。現有 FP32、batch1..4、clip1..90
與 execution lock 政策保持不變。

## 可重現命令

在 repository root 執行，真硬體程序需要可存取 Metal GPU 的 host 環境。

```bash
PY312=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
PY313=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-313/bin/python
WEIGHTS='/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights'
EVIDENCE=/Users/kaho/jasna-mac/verification-evidence/issue31
"$PY312" --version
"$PY312" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(),torch.backends.mps.is_available())'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize
ffmpeg -version
ffprobe -version

"$PY312" -m scripts.profile_mps_models --weights "$WEIGHTS" \
  --output "$EVIDENCE/models.json" --runs 3 --clips 2 16 45 90

# 各 run 使用不同、尚未存在 output 的目錄；依序執行以免 GPU 互相干擾。
# 短片控制組：同一命令分別省略／加上 --observe，以量化 observation overhead。
JASNA_ENCODE_BACKEND=software JASNA_DECODE_BACKEND=pyav-sw \
"$PY312" -m scripts.profile_mps_pipeline \
  --input assets/test_clip1_1080p.mp4 --weights "$WEIGHTS" \
  --output-dir "$EVIDENCE/fixture-control" --eager-detector

JASNA_ENCODE_BACKEND=software JASNA_DECODE_BACKEND=pyav-sw \
"$PY312" -m scripts.profile_mps_pipeline \
  --input /Users/kaho/jasna-mac/test_clip.mp4 --weights "$WEIGHTS" \
  --output-dir "$EVIDENCE/long-before" --eager-detector --observe

JASNA_ENCODE_BACKEND=software JASNA_DECODE_BACKEND=pyav-sw \
"$PY312" -m scripts.profile_mps_pipeline \
  --input /Users/kaho/jasna-mac/test_clip.mp4 --weights "$WEIGHTS" \
  --output-dir "$EVIDENCE/long-after" --observe
```

E2E 固定 batch4／clip90／overlap8／FP32；使用 software H.264，以保持 #12 的 media
品質／模型設定，並避免混入另一個 encoder 變數。#15 的硬體路徑已獨立驗證，這次沒有
將其收益歸給 inference optimization。每個 run 保存 input SHA256、weights SHA256、
實際 dependency versions、完整 CLI argv、frame/detection/restoration counts、記憶體
samples、ffprobe 與完整 decode 結果。weights 前後 hashes 必須一致。

## 如何讀取量測

`measurements.json` 的 stage wall time 是 **CPU thread 的 inclusive／exclusive wall time**，
不是 GPU kernel profiler。一般 E2E observation 不新增 `torch.mps.synchronize()`；只量測
原有 completion。`*_submission` 可能在 GPU 完成前返回，後面的 postprocess／copy／鎖退出
可能承擔等待，不能把這些數字直接當作 kernel costs。

`lock.hold/<thread>` 僅計 outermost acquisition，包含原有 completion；各 thread 的 hold
互斥，可用於 serialized critical-path decomposition。`lock.wait` 是 competing workers
的等待；不能把各 thread 的 waits 相加當作可消除 wall time，也不能認為取消鎖便可獲得
該等待時間的 speedup。沒有移除 correctness lock 的 production experiment。

LoopTimer rows 分開列出 queue wait、decode、detect/track、restoration、blend、write。
`media.encode_and_pack` 的 exclusive time 是 PyAV frame packing＋codec encode；其 child
`media.rgb_yuv`、`media.mux_video` 可分開讀取。`media.audio_copy` 包含 source packet pumping。
Decoder colorspace 由 PyAV reformatter 邊界量測；packet decode 是另一個邊界。
跨裝置 `to`／`cpu`／`copy_` 的 units 為來源 tensor bytes，含 call count 與 bytes/s；
這是 application API boundary 統計，不是實體 bus traffic，沒有計入 native op 內部或
driver 的隱式 copies。encoder snapshot 的原有 blocking copy 有自己的 transfer row。

`profile_mps_models` 是獨佔 GPU、單一 Python thread 的 isolated experiment，明確在
stage 邊界同步；cold call 與三次 warm samples 分開保存。BasicVSR++ propagation timing
包含 alignment child，exclusive 欄才排除 child。逐個 alignment 同步會增加 overhead，
需與未 instrument 的完整 forward 對照，不能直接外推 E2E。
每種 precision 之前從保留的原始 FP32 state 嚴格還原，避免 half→float 權重舍入污染
後續對照組。FP16／autocast 僅實驗，不更改 production precision policy。

Amdahl 的條件式上限為 `S = 1 / (1 - f + f/s)`；若整個 stage 完全免費，
`S_max = 1 / (1-f)`。本次以不重疊的 outer lock hold／E2E wall 估算 serialized stage
份額。這是理想化上限；不是取消 queue waits 或單獨 kernel 的實測收益，亦不保證
memory／submission bottleneck 在優化後保持固定。

## 外部參考

檢查 [ladaapp/lada](https://github.com/ladaapp/lada) commit
`20cb34a20a83c72c87a991d2c949032c70085b16`：MPS FP16 capability、BasicVSR++ `.half()`、
PyAV `thread_type=AUTO`、VideoToolbox encode、MPS flow_warp workaround 均有 source
依據。其預設 detector 為 v4-fast，不能與 RF-DETR v6 的 workload／quality 當成等價。
沒有取得可重現的同機同片公開 FPS 證據。

`https://github.com/horuke-rere/ladamac.git` 的 clone 回應 `Repository not found`。
目前無法重現使用者回憶中的數十 FPS；不將這項回憶當成性能 ceiling 或 backend parity
證據。保持其比較價值，並明確記錄未重現的限制。

## 測試及實測結果

完整結果與 acceptance checklist 見同 PR 的 `Codex M2 Pro Verification` comment。
Focused tests 保留原有 CPU reference、finite、device、shape 與 boxes／masks assertions，
新增 batch1／2／3／4 的 eager/export 精確一致性、tuple→public dict contract、CPU-first
export 與 NVIDIA／AMD 不啟用 optimization 的 guards，以及 profiling 失敗後完整還原。
