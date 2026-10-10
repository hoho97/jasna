# Apple/MPS throughput profiling（Issue #31）

機器可讀的完整摘要：[mps-throughput-results.json](mps-throughput-results.json)。
包含 exact verification argv／exit codes、dependency versions、weights hashes、stage
calls／wall／rates、memory summaries、編譯探測與 decoded-frame comparison。

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

候選 optimization（明確 opt-in，預設不啟用）：Apple/MPS 使用 rfdetr 1.8.3 已有的
`optimize_for_inference(compile=False, dtype=torch.float32, inplace=True)`。
在 CPU 完成 checkpoint 載入及 export，再移到 MPS；省去一般 forward 計算、但 Jasna
不消費的 auxiliary／encoder segmentation outputs。三個公開 tensors 的名稱、shape、
FP32 與 device contract 保持一致。不新增 compile、模型替換、quality reduction、CPU
model fallback 或第三方套件全域 monkey patch。

`JASNA_MPS_RFDETR_EXPORT=1` 明確選用候選 export；未設定時保留原有 MPS eager forward。
benchmark script 的 `--eager-detector` 選原有路徑，省略則選 export。
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
  --output "$EVIDENCE/models-final.json" --runs 3 --clips 2 16 45 90

# 各 run 使用不同、尚未存在 output 的目錄；依序執行以免 GPU 互相干擾。
# 短片控制組：同一命令分別省略／加上 --observe，以量化 observation overhead。
JASNA_ENCODE_BACKEND=software JASNA_DECODE_BACKEND=pyav-sw \
"$PY312" -m scripts.profile_mps_pipeline \
  --input assets/test_clip1_1080p.mp4 --weights "$WEIGHTS" \
  --output-dir "$EVIDENCE/fixture-control" --eager-detector

ffmpeg -v error -i /Users/kaho/jasna-mac/test_clip.mp4 -t 120 \
  -map 0:v:0 -map '0:a?' -c copy "$EVIDENCE/benchmark-first120.mp4"

JASNA_ENCODE_BACKEND=software JASNA_DECODE_BACKEND=pyav-sw \
"$PY312" -m scripts.profile_mps_pipeline \
  --input "$EVIDENCE/benchmark-first120.mp4" --weights "$WEIGHTS" \
  --output-dir "$EVIDENCE/prefix-before-detail" --eager-detector --observe --detail-transfers

JASNA_ENCODE_BACKEND=software JASNA_DECODE_BACKEND=pyav-sw \
"$PY312" -m scripts.profile_mps_pipeline \
  --input "$EVIDENCE/benchmark-first120.mp4" --weights "$WEIGHTS" \
  --output-dir "$EVIDENCE/prefix-after-detail" --observe --detail-transfers
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

另已 clone 使用者提供的 [Codeberg ladamac](https://codeberg.org/horuke-rere/ladamac)
commit `7c2d41ed38d1b8984e9f3f2d66465be61ec05c5b`。該 checkout 包含 GUI／打包程式，
沒有 `lada.lib.frame_restorer`、`mps_utils`、`macos_utils` 核心或 submodule；`LADA.spec`
從 checkout 以外的 parent source 目錄匯入。因此無法僅憑這份公開 source 重現其完整
inference pipeline，也不將其他 Lada 版本冒稱為 ladamac。

唯讀 release weights 沒有 ladamac 的 v3.1-fast；現有 v4-fast 僅用於 detector 成本分解。
GUI 可確認使用 YOLO `v3.1_fast`、BasicVSR++ generic、clip180、VideoToolbox／software
encoder 選擇與結束時同步／清 cache。`CHANGELOG.md` 的 1.2.0 描述等長 clip batching
（32 GB 可 batch3），宣稱 restoration 從 39 降到 18 ms/frame；這是作者的測量，沒有
提供可在此 checkout 執行的核心與 benchmark，所以不是本機重現結果。由此補做現有
BasicVSR++ 的 N=1／2／3 等長 clip90 isolated experiment；沒有提前擴大 #30 的 clip 上限。

`DEVELOPMENT_NOTES.md` 列出全域 MPS fallback 與停用 high watermark 的設定；本次
不沿用這些設定。ladamac 的實際 detector batching／cadence、precision、crop／overlap
與核心 synchronization 無法由 GUI 證實。不能用其數十 FPS 的回憶或不同 detector 的
benchmark 證明 RF-DETR v6 的品質等價收益，亦不能據此認定 Apple backend 已到上限。

## 測試及實測結果

完整結果與 acceptance checklist 見同 PR 的 `Codex M2 Pro Verification` comment。
Focused tests 保留原有 CPU reference、finite、device、shape 與 boxes／masks assertions，
新增 batch1／2／3／4 的 eager/export 精確一致性、tuple→public dict contract、CPU-first
export 與 NVIDIA／AMD 不啟用 optimization 的 guards，以及 profiling 失敗後完整還原。

最終回歸：378 passed／1 skipped／4 deselected（18.43 s）；Python 3.12／3.13 的
real focused suites 各 30 passed（33.83／33.35 s），包含兩種 RF-DETR checkpoint 的
eager／export CPU reference。Python 3.13 profiling／routing／timing units 為 27 passed；
真實 VideoToolbox CLI tests 2 passed。skip 為需要另一 GPU backend 的既有測試，
不是 Apple model inference skip；四項 `mps_real` loader tests 另由 marker 排除，
不能算成通過。完整 argv 與結果保存在 JSON `tests` 欄位。

最後新增 detail observation 的 public fixture control 為 51.492550 s，observed
51.715230 s，wall overhead +0.432%；原有 stage observation pair 的 overhead 為
-0.377%／+2.810%。這是單次 pairing 的估計，並非穩定誤差界。

warnings：rfdetr checkpoint 的 query-count inference／deprecated API、TorchScript
trace warnings、Inductor autotune 提示，以及 PyAV／OpenCV bundled libavdevice 的
`AVFFrameReceiver`／`AVFAudioReceiver` duplicate-class warning；未觀察到相關 crash。
不為此變更本 Issue 的 dependency versions，亦不把沒有 crash 當成消除該警告。

### 已完成的完整長片對照（保留為參考）

| 同片、同模型 FP32 | Wall | FPS | 相對 eager |
| --- | ---: | ---: | ---: |
| 原有 eager | 5541.134 s | 5.38861 | 1.000x |
| inference export | 5672.299 s | 5.26400 | 0.97688x |

兩者均為 29859 frames、7465 detector calls、20431 positive frames、21268 detections、
496 restoration calls、24560 input crop frames、22100 kept crop frames。ffprobe、完整
PyAV／FFmpeg decode 通過；`-fps_mode passthrough -f framemd5` 的 29859 個 decoded
frame hashes 完全相同。原始 weights SHA256 前後相同。**這組結果沒有性能改善**。

public 300-frame fixture 的無 observation 對照為 5.81668 → 6.14208 fps（+5.59%），
與完整長片的方向不同。observation run 為 5.83869／5.97420 fps；相對各自 control，
wall 為 -0.38%／+2.81%。單次 pairing 無法區分小幅收益與測量波動，不能只挑短片
數字宣稱長片加速。

每 5 秒 sampling 的 driver allocation quarter means，eager 為
7.742／9.554／10.656／12.544 GiB，export 為 6.817／9.446／10.030／11.487 GiB，
兩者均有 cache／allocation growth。sampled peaks（eager／export）：driver
14.216／13.388 GiB、current allocated 2.665／2.610 GiB、RSS 6.999／7.160 GiB、
host swap 3.495／2.343 GiB。swap 是整台 host 的數值；起始 available memory 也不同，
不能把 swap 差異歸因於 patch，亦未量到 5 秒間隔以外的瞬時 peak。

依使用者後續指示，後續 optimization trials 固定使用長片首 120 秒，以 stream copy
保留原始畫面：實際 3597 frames、video duration 120.020 s、PTS 0..119.986533 s，
嚴格遞增；SHA256 `e1b28346612e962d72366b1f080e9dde9b264bfd6f176ffed5591ce42f0fa8a2`。
不再重跑完整長片作候選試驗。完整長片結果仍保留作 workload／memory 參考。

| 固定首 120 秒、同 FP32 設定、同 detail observation | Wall | FPS |
| --- | ---: | ---: |
| 原有 eager | 713.323 s | 5.04260 |
| inference export | 687.913 s | 5.22886 |

相對 +3.69%；兩者 counts 均為 900 detector calls／3597 frames／2762 positive frames／
2923 detections／45 restoration calls／3485 input crop frames／3065 kept crop frames。
ffprobe、完整 decode、FP32／MPS／finite 與 weights unchanged checks 全部通過。
before／after 的 3597 個 decoded frame hashes 完全相同；stream-copy 片段也與來源
首 3597 幀的 decoded hashes 完全相同。
仍未達最低 >6 fps，且完整長片 pairing 沒有同方向改善，故保留 eager 預設。

### Isolated experiments

以下為 `models-final.json` 的三次 warm median；cold 與所有 samples 另存 JSON。

| RF-DETR batch | 原 forward FP32 | export FP32 | export 的 raw outputs |
| --- | ---: | ---: | --- |
| 1 | 0.127894 s | 0.116814 s | exact |
| 2 | 0.227950 s | 0.212836 s | exact |
| 4 | 0.455870 s | 0.454778 s | exact |

batch4 的最新 isolated pairing 沒有明顯 export 收益；早先 pairing 約 8% 的收益
不是穩定保證。BasicVSR++ FP32 clip2／16／45／90 的 median 分別為
0.059179／0.475189／1.327927／2.681434 s。autocast 在長 clips 出現 NaN；pure FP16
有限樣本 finite，但收益小且會改變輸出。這些 precision paths 均不作 production policy。

N=1／2／3、T=90 的等長 restoration batching 為 33.605／36.705／40.697 crop fps；
N=3 相對 N=1 約 +21.1%，對各自獨立 FP32 clips 的 max error／RMSE 均為 0。
N=3 forward 後 driver allocation 約 10.28 GiB（包含本模型試驗的 cache，並非 peak）。
套用 baseline 的 restoration 份額，理想 E2E 收益約 3.2%，沒有重現 ladamac 的 2 倍
restoration claim，也沒有因此修改 pipeline scheduler。

既有 scan-only CPU area fallback 的 batch4 144×144→37×65 探測，download／CPU
resize／upload median 約 0.160／0.072／0.210 ms；與 CPU reference exact。普通 E2E
沒有 scan calls，故該 fallback 在這些 E2E runs 的成本為 0。YOLO v4-fast 的 public
batch4 detector 約 93.94 fps，僅作不同 detector 成本分解，並非 RF-DETR quality parity。

### Compilation feasibility

Inductor 原始探測失敗：rfdetr 的 private `torch._shape_as_tensor` 在 Dynamo capture
成為 shape tuple，後續 `torch.stack` 發生 TypeError。`suppress_errors=False`，沒有
靜默退回 eager；沒有成功的 warm throughput／numerical result。
最終 fresh-cache 重跑的 failed first execution 為 5.667 s，explain attempt 1.587 s；
整個探測（含模型載入與隔離 handoff check）11.998 s，結果仍為 FAILED。

TorchScript trace 花 3.940 s；batch4 cold execution 0.569299 s、三次 warm median
0.421879 s，對同程序 exported eager median 0.437987 s 僅約 +3.8% throughput。
raw outputs、selected boxes／masks 與 partial batch1／2／3 完全一致。trace 仍有
tensor→Python boolean／iteration、deprecated API warnings，graph 含 PythonOp；
不能視為完整無 graph break 的可攜 graph，也未默認啟用。
最終重跑 trace 3.585 s、cold execution 0.533608 s、warm median 0.396246 s，對同程序
eager 0.403591 s 僅 +1.85%；partial batch1／2／3 warm median
0.112785／0.202450／0.300613 s，raw outputs 仍 exact。沒有只挑較有利的初次 +3.8%。

第二個 Inductor 探測僅在獨立程序暫時將 private shape op 換成 static CPU tensor
construction，與原有 forward exact parity 後才編譯。此 workaround **不安裝到 application**。
得到 13 graphs／12 graph breaks（data-dependent scalar、deform sampling tensor iteration、
segmentation flags context）；cold 10.673 s、warm median 1.277060 s，較同程序 eager
0.430335 s 慢約 3 倍。batch1／2／3 的 cold 分別 8.341／8.705／8.937 s，warm median
0.315107／0.639049／1.020538 s。FP32 finite，batch4 selected boxes max error
0.004883 pixels、selected masks exact，但因性能退步而拒絕；沒有拿數值通過代替性能通過。

第三個 Inductor 探測只編譯 backbone，完全使用標準 `torch.compile` API，沒有 shape
monkey patch：1 graph／0 graph breaks，cold 5.485 s、warm median 0.916113 s，對同程序
eager 0.414320 s 仍慢 2.21 倍。partial batches 1／2／3 也完成，各自 warm median
0.238554／0.464026／0.687013 s；selected masks exact、selected boxes max error
0.004883 pixels，但同樣不採用。這也說明僅移除 graph breaks 並不足以保證 MPS 收益。

### Transfer wall attribution

完整長片 eager／export 的 CPU→MPS API boundary wall 為 52.543／3058.080 s，
calls 與 bytes 卻同為 23783／371714994148。新增 optional `--detail-transfers`
按 thread／dtype／shape 歸屬，指出 export 的等待集中在 DecodeDetect 的 int64 `(1,2)`
shape tensor，而非影像 payload。detail rows 是 aggregate transfer 的 child，不能再相加。

獨立實測同樣 16-byte CPU→MPS blocking copy：GPU 已完成時 median 0.165 ms；
剛提交真實 RF-DETR backbone 後需 389／437／499 ms，而 backbone 的 Python submission
只需約 7.2–7.4 ms。故 blocking copy wall 包含前序 GPU completion。兩路徑的等待歸屬
不同，不能將 3058 s 認作 bus copy 成本或直接加到 forward wall；把等待搬到另一個
boundary 也不能算作加速。沒有因此更改 blocking handoff／buffer ownership 政策。

固定兩分鐘進一步驗證：DecodeDetect 的 900 次 shape copies（共 14400 bytes）在
eager／export 分別佔 3.060／353.518 s；同一 reader 約 22.38 GB 的 image uploads
則只佔 1.553／1.259 s。這與隔離等待實驗一致，而非影像上傳突然變慢。

## Critical path 解讀與候選取捨

同片 eager baseline 為 5541.13 s／5.389 fps；不可將 #12 的歷史 4.426 fps
直接當成此 patch 的 before。不同 system load／cache／版本可能影響歷史數字。

| 互斥 outer lock hold | Baseline wall | E2E 份額 | 該區域快 2 倍的理想 E2E speedup | 該區域免費的理想上限 |
| --- | ---: | ---: | ---: | ---: |
| DecodeDetect | 4168.67 s | 75.23% | 1.603x | 4.037x |
| PrimaryRestore | 992.35 s | 17.91% | 1.098x | 1.218x |
| BlendEncode | 366.28 s | 6.61% | 1.034x | 1.071x |

這些是 **條件式** Amdahl bounds，不能相加；DecodeDetect 也包含 transfers／tracking
與完成等待，並非全部能由 RF-DETR forward optimization 消除。queue／lock wait rows
是跨 thread 重疊的等待，僅用於判斷 starvation／contention。

原有 PrimaryRestore queue wait 約 4461.04 s，BlendEncode lock wait 約 5088.22 s；
這反映 detector producer／MPS serialization 的等待，不能把等待相加成可回收成本。

| 完整長片 eager boundary | Calls | Inclusive wall | Exclusive wall | Calls/s |
| --- | ---: | ---: | ---: | ---: |
| decode packet（兩個 readers） | 60090 | 258.789 s | 258.789 s | 232.20 |
| decode colorspace | 59718 | 20.040 s | 20.040 s | 2979.94 |
| RF preprocess | 7465 | 2.441 s | 2.441 s | 3058.75 |
| RF forward submission | 7465 | 3217.373 s | 3191.739 s | 2.32 |
| RF postprocess | 7465 | 738.000 s | 730.656 s | 10.12 |
| restoration crop preparation | 496 | 114.427 s | 114.427 s | 4.33 |
| RGB→YUV | 29859 | 188.231 s | 188.231 s | 158.63 |
| packing＋encode（含 children） | 29859 | 599.620 s | 406.149 s | 49.80 |
| mux（含 audio pumping child） | 29859 | 5.243 s | 1.073 s | 5694.50 |
| audio packet pumping | 29860 | 4.176 s | 4.176 s | 7150.72 |

這些 wall rows 並非互斥，且 asynchronous submission／encode worker 可重疊。
JSON 保留未四捨五入的每項 call rates、CPU bytes/s、exclusive wall 與完整 queue／lock rows。

- RF-DETR inference export：移除不消費的輸出，維持 FP32；E2E 收益未穩定重現，
  僅保留明確 opt-in，不更改預設。
- RF-DETR native grid sampling：isolated sampling op 快很多，但整個 model 的收益很小；
  需要修改外部套件或全域 patch，不採用。
- RF-DETR FP16：需 detection quality gate；有限 sample 的 selected boxes／masks
  已有差異，不據此更改既有 FP32 policy。
- BasicVSR++ autocast：長 clips 出現 non-finite，拒絕。pure FP16 有限樣本 finite，
  但收益小且未有完整品質 corpus，不採用。
- 等長 restoration batching：只做 N=1／2／3、T<=90 的模型實驗；不改 scheduler
  或 #30 上限。其 E2E 收益受 PrimaryRestore 的 17.91% 份額限制。
- 鎖縮小／取消：wait 不能視為可直接回收成本；既有 serialization 是實際 Metal
  command-buffer correctness 修復，沒有以本次量測證明可安全移除。
- Blocking copies／preallocation：baseline 直接 API transfer wall 很小；export 的
  CPU→MPS boundary wall 卻大幅增長。這是等待歸屬的待查項，不能當作實體 bandwidth
  或可直接回收的成本；沒有改變 owned snapshot 與非同步生命週期。
- torch.compile／TorchScript：做獨立 feasibility probe，記錄失敗／cold／warm／
  graph breaks／numerical parity；未默認啟用。Core ML／Metal rewrite 不在此次實作，
  亦不宣稱目前已達 Apple hardware/backend 上限。

候選優化還可用基線 boundary wall 計算較寬鬆的個別上限（假設該區間全在
critical path 且其餘工作固定）：RF forward submission 的 f=58.06%，免費上限
2.384x；CPU→MPS API transfers 的 f=0.948%，上限 1.010x；MPS→CPU API transfers
f=0.721%，上限 1.007x；原有 completion wait f=0.730%，上限 1.007x。
這些區間嵌在 outer lock／其他 stages 中，不能彼此相加。尤其 asynchronous GPU
成本可能歸入後續 postprocess／copy；forward submission bound 不能視為完整模型
kernel bound。任何 compile／shape／precision experiment 都需自己的實測收益。

以該 baseline RF boundary f=0.5806、restoration hold f=0.1791 代入候選實測倍率：

| 候選 | Isolated 倍率 | 條件式理想 E2E 倍率 | 取捨 |
| --- | ---: | ---: | --- |
| TorchScript（最終 pairing） | 1.0185x | 1.0107x | 收益小，不默認啟用 |
| RF gather FP16（最終 matrix） | 1.0706x | 1.0398x | boxes／masks 改變，拒絕 |
| restoration N=3 vs N=1 | 1.2110x | 1.0322x | 只做模型探測，不擴大 scheduler scope |
| 去除全部 overlap work | 假設 13.36% restoration work 免費 | 1.0245x | 會改變 temporal context，不採用 |

完整模型／backbone-only Inductor 的 isolated 倍率均低於 1，已在模型層拒絕，沒有
再花一個 E2E run 驗證已知退步。native sampling 的整模型收益在不同 pairing 中
波動，最新 baseline 0.455870 s 對 native FP32 0.454429 s，沒有高收益證據。

Windows TensorRT 的約 150 fps 包含不同 hardware、FP16 engine、CUDA streams 與
hardware codecs；本次只有 M2 Pro FP32 Torch 的同片對照。RF-DETR batch4 本身的
isolated throughput 約 10 fps，單靠換 encoder 無法達到 15–20 fps。這是目前模型
路徑的實測成本，不是 Apple Silicon 的固有上限；YOLO decomposition 也不構成品質
等價替換。沒有實機 NVIDIA／ROCm／Windows／Linux hardware 性能驗證。

## Temporal overlap 成本

基線實際送入模型 24560 crop frames，primary keep ranges 共 22100 frames，故直接量測
到丟棄 context 2460 frames（input work 的 10.02%）。這本身只是 overlap 下界，因為
crossfade 中保留的 frames 仍可重複運算。

本命令 crossfade 預設開啟；`pipeline_threads` 使用 d=8、bf=floor(8/3)=2。
`pipeline_processing` 每次 max-size split 複製 2d=16 raw crops，`compute_keep_range`
在 parent/child 各捨棄 d-bf=6 frames，所以一個完成的配對 split 丟棄 12 frames，
並保留 4 個重複 crossfade frames。以完整完成、沒有 clip cancellation 的配對規則推導：
2460/12=205 splits，重複 raw work=205*16=3280 frames，去重後 tracked crop work
約為 21280 frames；overlap 佔 raw work 13.36%，相對去重後工作量多 15.41%。
這是由 counts 與現行 split semantics 推導，沒有直接逐 scene 儲存 unique-frame IDs；
不同 tracks 同一畫面各自的 crop 不應當成 temporal overlap。

若假設所有 overlap work 都可免費且其餘固定，以 PrimaryRestore hold 份額估算的
理想 E2E speedup 約 1.025x。實際不能直接移除 temporal context／crossfade 來宣稱
品質等價收益，因此本次保留 overlap8、clip90，沒有擴大 #30 scope。

## Acceptance 狀態

結果為 **PARTIAL**。已實測 export、native sampling、兩個模型的 precision、等長
restoration batching、TorchScript、完整模型 Inductor（含相容性修正）、backbone-only
Inductor；均未證明符合品質門檻的高收益 E2E 改善。保留 eager 預設，沒有默認開啟
較慢或品質不同的方案。這不等於證明 Apple backend 已接近其實際上限。

- [x] #15 與既定 prerequisites 在 base。
- [x] 固定模型／weights／input／settings 的可重現 baseline 與 paired comparisons。
- [x] 主要 stages wall／calls／rates、observation overhead 與測量限制。
- [x] top bottlenecks、Amdahl conditional bounds、lock hold／wait／queue starvation。
- [x] RF-DETR batch1／2／4、FP32／FP16／autocast 與 numerical／selected detection gates。
- [x] BasicVSR++ clip2／16／45／90、FP32／FP16／autocast 與 finite gates。
- [x] 編譯 cold／warm、graph breaks、partial batches 與 numerical results；無收益不默認啟用。
- [ ] 至少一項已證明高收益的 E2E optimization，或現有 backend 接近上限的證據。
- [x] before／after 維持相同模型與品質；後續使用使用者指定的固定首兩分鐘。
- [x] relevant regression tests、明確 scoped CPU fallback 與成本。
- [x] 說明 TensorRT／FP16／streams／hardware codecs 與 MPS FP32 的不可直接等同比較。

最低 >6 fps、可接受 >=10 fps 與理想 15–20 fps 目標均未達成。ladamac 的完整同機
重現另外受缺少公開 inference core 與 v3.1-fast checkpoint 所阻；現有 YOLO v4-fast
探測只作不同模型成本分解。沒有實機 NVIDIA／ROCm／Windows／Linux 性能覆蓋。
