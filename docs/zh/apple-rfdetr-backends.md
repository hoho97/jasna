# RF-DETR v6 Apple-native backend（Issue #36）

此功能以明確選擇的 MLX／Core ML backend 加速 Apple Silicon 上的 RF-DETR v6。
預設仍為 PyTorch MPS eager；NVIDIA／AMD／Windows／Linux 的路由不變。
BasicVSR++、576 resolution、200 queries、threshold 0.35、batch4／clip90／overlap8
及 encoder/media 設定保持一致。只支援 v6 medium；其他 detector 可使用原有 Torch 路徑。

## 選擇及安裝

在 macOS arm64 的專案環境安裝對應 extra：

```bash
python -m pip install -e '.[macos,macos-mlx]'
JASNA_APPLE_RFDETR_BACKEND=mlx jasna --device mps --input input.mp4 --output output.mp4

# 明確返回既有 eager fallback
JASNA_APPLE_RFDETR_BACKEND=torch JASNA_MPS_RFDETR_EXPORT=0 \
jasna --device mps --input input.mp4 --output output.mp4
```

MLX 固定 0.31.2，所有運算 FP32，使用 fused scaled-dot-product attention。
不使用 `mx.compile(model)`：此版本在真正 CLI worker 結束時曾於 CompilerCache 的
TLS teardown 觸發原生 `dict_dealloc` segmentation fault；單執行緒 benchmark 不會暴露它。
穩定版本保留 fused SDPA eager，沒有改動 #37 的 execution lock／concurrency policy。

Core ML 需事先匯出，不能直接使用 COCO 模型或其他 checkpoint 的 package：

```bash
python -m pip install -e '.[macos,macos-coreml]'
git clone https://github.com/roboflow/rf-detr /tmp/rf-detr-coreml
git -C /tmp/rf-detr-coreml checkout eca736acab9f6fe93f2cbc82a1d9f6edfb53f0ea
python -m scripts.export_rfdetr_coreml --weights /path/to/rfdetr-v6.pt \
  --official-source /tmp/rf-detr-coreml --output-dir /path/to/coreml-v6
JASNA_APPLE_RFDETR_BACKEND=coreml JASNA_RFDETR_COREML_DIR=/path/to/coreml-v6 \
jasna --device mps --input input.mp4 --output output.mp4
```

Exporter 在獨立程序匯入指定 official checkout，不替換已安裝的 `rfdetr==1.8.3`。
先對原始 state dict 執行 `strict=True` 載入，再使用官方 Core ML exporter 產生 FP32
static batch 1／2／4；batch3 以最後一張補到4，輸出裁回3。Manifest 綁定 checkpoint
SHA256、precision、resolution／queries／classes／mask shape，拒絕缺失或外部模型路徑。
執行使用 `CPU_AND_GPU`；沒有宣稱此 FP32 graph 在 ANE 執行。
Batch4 ComputePlan 在 `CPU_AND_GPU`／`ALL` 都回報818個 operations 的 preferred
device 為 GPU、supported devices 為 CPU／GPU；其餘1262個為不提供 placement 的
`const`。沒有列出 ANE。這是官方預期 placement API，不是逐 kernel runtime trace。
CPU_ONLY／CPU_AND_NE 的 FP32 batch4 median 約832／821 ms，GPU／ALL 約287／285 ms。
另測官方 FP16／ANE rewrite：batch4 GPU約389 ms、ALL約416 ms，boxes誤差最高8.45 pixels、
mask IoU最低0.862、scores誤差最高0.043，故排除。CPU_AND_NE loading超過兩分鐘，
已終止該獨立實驗；沒有據此宣稱 ANE graph 可用。

舊 stack 的直接 conversion 曾遇到 tracing／data-dependent assert、private shape
欄位及 rank6 deformable sampling graph 問題。指定的官方 exporter 已提供 torch.export
與 rank≤5 deformable attention rewrite／op aliases，完整 custom v6 graph 可轉換，
因此沒有加入 ONNX 中轉、hybrid crossing 或 custom Metal kernel。官方 dynamic batch
仍未支援，使用固定1／2／4。Coremltools9.0 亦警告 Torch2.12 超出其官方測試範圍
（最新官方已測2.7），所以固定版本、實機 parity 與 E2E 驗證都是必要限制。

## Checkpoint 與架構審查

唯讀 release `rfdetr-v6.pt` SHA256：
`f10bedc4d105c2721e4259b8680203d51f344f73e55e85710d915619f5731b55`。
實際 checkpoint 為 573 tensors，DINOv2 small backbone（384 dim／6 heads／12 layers／
patch12／2 windows），projector P4、features 3／6／9／12，hidden256、5 decoder layers、
self attention 8 heads、deformable attention 16 heads／2 points、200 inference queries、
2600 grouped training embeddings、3 logits channels、segmentation output 144×144。
原始 args 缺少部分 architecture 欄位，因此 adapter 同時依據實際 names／shapes 和
既有 Torch loader 的建構行為，沒有直接套用 fork 的 seg-medium432／seg-large504。

Scoped MLX inference graph 來自 [Ak-Gautam/rf-detr-mac](https://github.com/Ak-Gautam/rf-detr-mac)
`939362b452d317807c130ae2c82610c564ce2fe2`。審查 config、convert、backbone、model、layers、
ops、postprocess、兩個 MLX test files，以及 inference port／sampling 修正的歷史。
其 tests 主要涵蓋 config／tree／layout，不能代替實際 checkpoint parity。
沒有匯入高階 PIL predictor、訓練程式、下載或該 fork 的 dependency stack。
Apache-2.0 來源及修改記錄見 `jasna/mosaic/mlx_rfdetr/NOTICE.md`。

Conv 權重 OIHW→OHWI；transposed projector conv IOHW→OHWI；linear／LayerNorm
保持原布局。Deformable sampling 以 NHWC 四鄰點 gather 實作 zero-padding、
`align_corners=False`，其餘圖沿用 NCHW contract。Segmentation head／projector／
decoder／class head 都納入 strict name-and-shape validation；任何 missing、unexpected、
shape mismatch 一次完整列出。空 `_kp_active_mask[0,0]` 明確映射到 MLX 可列舉的
`kp_active_mask`。MLX safetensors 不支援空 tensor，因此 audit conversion 將它記於
metadata 並重建後再 strict load，沒有忽略 checkpoint 內容。

關鍵 compatibility 修正：既有 Torch CPU-first loader 在覆寫 resolution576 時，
會先將 checkpoint 的 36×36 positional embeddings 以 antialiased bicubic 轉到48×48。
原 fork runtime resize 的非 antialiased 結果不同。MLX runner 使用相同 CPU transform
一次產生位置 cache，保留原始36×36參數作 strict validation；沒有改變 inference quality。

## 交接與同步

Torch2.12 可產生 Metal DLPack capsule，但已安裝的 MLX0.31.2 沒有 `from_dlpack`，
`mx.asarray(MPS tensor)` 亦回報 `std::bad_cast`。MLX main branch 文件中的新 API
不能視為此已安裝版本的能力。本版沒有使用私有 Metal storage pointer。

流程為既有 Torch MPS resize／normalize → blocking owned FP32 CPU snapshot → MLX
array → native forward／top16／threshold／mask>0 → 只將選取 bool144×144 masks 送回MPS，
boxes 留在 CPU。`mx.eval` 完成輸出相依 graph；NumPy owned copies 保持跨框架生命週期。
沒有新增全域 MPS synchronize。原始 `infer`／scan API 仍提供完整 FP32 raw tensors，
因此 scan 不享有 selected-output handoff 的全部收益。
Core ML 也沿用 blocking input，原生 runtime 返回 raw NumPy outputs，CPU 選取後只回傳
選取 bool masks；其 raw mask 回傳成本已包含在 benchmark，沒有假設 zero-copy。

## 可重現驗證

```bash
python -m scripts.convert_rfdetr_mlx --weights /path/to/rfdetr-v6.pt \
  --output /path/outside-weights/v6.safetensors
python -m scripts.benchmark_rfdetr_apple --weights /path/to/rfdetr-v6.pt \
  --input /Users/kaho/jasna-mac/test_clip.mp4 --frames 300 900 1800 2700 \
  --runs 7 --output /path/to/isolated.json
JASNA_TEST_APPLE_BACKENDS=mlx,coreml JASNA_TEST_MODEL_WEIGHTS_DIR=/path/to/model_weights \
JASNA_RFDETR_COREML_DIR=/path/to/coreml-v6 \
python -m pytest -q -s tests/test_rfdetr_apple_real.py
```

Isolated benchmark 僅在獨佔 GPU 的單一 Python thread 於計時邊界同步；保存 cold／warm
samples、preprocess、raw forward＋handoff、完整 detector public call、memory、finite、
repeatability、raw max/RMSE 和 selected parity。產品 gate 為 count／positive decisions
相同、boxes≤1 pixel、mask IoU≥0.995、top16 scores 誤差≤0.001，threshold 嚴格 `>`。
不是要求浮點 raw bitwise 相等；低分 encoder proposal 排名接近時必須另行診斷／揭露。
長片第1800幀、batch4 的 encoder rank125／126（zero-based）在 MLX 交換；reference
分數僅差 `3.814697e-6`。原 raw boxes／logits／masks max error 為
0.03105／0.55499／25.96788；診斷程序只在 shadow forward 強制 reference proposal order，
三者降至 0.0001387／0.0017419／0.0211029（mask RMSE 0.0003500）。
這不是 production patch；實際選取結果仍以原始 native ranking 驗證，長片四張樣本
mask IoU=1、boxes error≤0.00049 pixel、scores error≤0.000051。

```bash
python -m scripts.diagnose_rfdetr_mlx_queries --weights /path/to/rfdetr-v6.pt \
  --input /Users/kaho/jasna-mac/test_clip.mp4 --output /path/to/proposal-order.json
```

`raw_with_handoffs.cold_seconds` 是模型首次 forward，包含 Core ML lazy load／compile；
其他欄位的 cold 是該 API 第一次量測，不能再當作未預熱的 model cold。
額外保存已完成 native outputs 上的 postprocess＋selected handoff，以及原生 forward
（Core ML 含其內部 NumPy input／output 契約）。Isolated RSS 同時含 resident reference
模型與 framework caches；真正 pipeline 的記憶體以 E2E samples 為準。

E2E 使用 PR #35 profiling infrastructure，各 backend 依序執行、獨佔 GPU：

```bash
# 固定同一 first120 streamcopy input、相同 flags，backend 分別 torch / mlx / coreml。
JASNA_APPLE_RFDETR_BACKEND=mlx JASNA_ENCODE_BACKEND=software JASNA_DECODE_BACKEND=pyav-sw \
python -m scripts.profile_mps_pipeline \
  --input /Users/kaho/jasna-mac/verification-evidence/issue31/benchmark-first120.mp4 \
  --weights /path/to/model_weights --output-dir /path/to/new-e2e-output \
  --eager-detector --observe --detail-transfers
```

`--eager-detector` 禁用既有 Torch inference export；native backend 由獨立 env 選擇。
記錄 counts、outer lock hold／wait、queue waits、CPU／AGX GPU counters、RSS／MPS／MLX
allocator memory、host available／swap、ffprobe、full decode、前後 weights SHA256。
Core ML 沒有相同公開 allocator API，RSS／AGX memory 是觀測值，不能冒充精確 Core ML
allocator peak。Stage inclusive／exclusive wall 仍不是 GPU kernel time，不可跨 thread 加總。

實際 M2 Pro 結果、完整 commands、限制和 acceptance 狀態見同 PR 的
`Codex M2 Pro Verification` 及 [apple-rfdetr-results.json](apple-rfdetr-results.json)。
只有120秒 E2E≥10 fps，才繼續完整長片；未達標時不宣稱 #31 usable-performance gate 已完成。

## M2 Pro 實測結果

環境：Apple M2 Pro／32 GiB、macOS27.0.1、Torch2.12、rfdetr1.8.3、
MLX0.31.2、coremltools9.0。下表為7次 warm detector public call 的 median，包含
前處理、input／selected-output handoff。兩次相同矩陣的 batch4 speedup 範圍為
MLX 1.55–1.61×、Core ML 1.56–1.64×；沒有達到優先的 batch4 2×。

| batch | PyTorch eager | MLX FP32 eager | Core ML FP32 CPU_AND_GPU |
|---|---:|---:|---:|
| 1 | 128.76 ms | 77.10 ms（1.67×） | 61.58 ms（2.09×） |
| 2 | 229.12 ms | 147.15 ms（1.56×） | 122.06 ms（1.88×） |
| 4 | 444.37 ms | 287.28 ms（1.55×） | 284.20 ms（1.56×） |

Batch4 原生 forward：MLX280.04 ms、Core ML predict275.11 ms（包含其 runtime
input／output 契約）；已完成輸出上的 postprocess＋selected handoff 分別2.49／1.47 ms。
Input blocking download／MLX array copy 分開保存在 JSON；當前 copy 成本不是倍數差距來源。

| 固定首120秒 E2E | wall | fps | 相對控制組 |
|---|---:|---:|---:|
| PyTorch eager | 698.483 s | 5.14973 | 1.00× |
| MLX | 483.820 s | 7.43459 | 1.44× |
| Core ML | 477.679 s | 7.53016 | 1.46× |

三組均為900 detector calls／3597 frames／2762 positive frames／2923 detections，
45 restoration calls／3485 crop frames／3065 kept crop frames，工作量完全相同。
三個輸出均通過 ffprobe、3597幀完整 PyAV decode、FP32 MPS restoration finite gate、
前後 weights hashes 一致。

Python3.12／3.13 relevant regression 各405 passed／1 skipped／4 deselected；
兩個版本 native raw／selected parity 與真 CLI detection→restoration→encode 測試各8 passed。
包含 positive／negative fixture、batch1／2／4；兩個 Python 的 selected boxes最大誤差
均為0.00757 pixel、mask IoU=1、top16 scores最大誤差0.0000707。
非 macOS native runtimes 不硬性匯入；NVIDIA／AMD guards 為 mock/routing regression，
沒有宣稱實際 NVIDIA／ROCm／Windows／Linux 硬體驗證。

**結論：#36 最低 detector≥1.5× 通過；#31 E2E≥10 fps 未達，整體 PARTIAL。**
沒有進行完整長片，也沒有開始 #37／#38。保留明確 opt-in 與 Torch eager fallback。
