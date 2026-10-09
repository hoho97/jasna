# PyAV software decode → MPS（Issue #10）

## Summary

為 `VideoReader` 新增 CPU RGB24 → blocking MPS upload 路徑。
Apple 的 `auto` 與 `pyav-sw` 使用真正 software decode，輸出 contiguous
uint8 BCHW 與原始 PTS；CPU 使用相同 host conversion 供 capability tests。

## Why

原 software path 使用 pinned YUV staging、private/current stream 及
`stream.synchronize()`；MPS 的 stream abstraction 回傳 None，不能直接沿用。

基準 `origin/feature/macos-mps`：`30c12d8`。已完整讀取 Issue #10、其 comments
及 Epic #1（兩者均無 comments）；直接 dependencies #3、#7 已 merge 且 closed。
origin 是 `hoho97/jasna`，upstream `Kruk2/jasna` 僅唯讀參考。

## Implementation

- 重用 PyAV `VideoReformatter` 的 RGB24 conversion，明確傳遞 matrix／range，
  丟棄 row padding，再 copy 到獨立、unpinned CPU BCHW batch。
- `host.to(device, non_blocking=False)` 為 handoff 邊界；沒有私有 stream、CUDA
  context、pinned async 假設或全域 CPU model fallback。
  [PyTorch Tensor.to 介面](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.to.html)。
- 每個 batch 擁有自己的儲存；停止 iterator 時不額外 read-ahead。
  沿用原 `_decoded_frames`／`_selected_frames` 的 PTS、seek、stride、corrupt-packet
  tolerance 及 EOF flush。取消維持 caller 在 batch boundary break／close 的 contract。
- CPU/MPS metadata 或實際 frame 的 >8-bit／PQ／HLG 明確拒絕。
  即使 metadata 漏報，仍檢查 frame depth／transfer tags。
- NVIDIA 的 VALI→PyAV escalation／NVDEC／private staging，及 AMD 的 AMF→software／
  batch YUV＋current stream 完全保留；沒有更改 Windows/Linux guards。
- 既有 seek tests 依 CPU、MPS、CUDA/ROCm availability 分類，原 speed／PTS assertions 保留。

Caller/callee review：`decode_detect_loop` 與 `blend_encode_loop`、GUI mosaic scan、
benchmark detection speed、LTX `video_frames` → `VideoReader.frames` → demux／
packet decode → seek filter → stride → batch → RGB24 → blocking upload。
`probe.get_video_meta_data` 提供 time_base/start_pts、matrix/range、depth/transfer。
檢查過 `YuvToRgbConverter` 的 NVIDIA kernel 與 AMD/MPS eager fallback；本 Issue 採用
建議的最短 RGB24 CPU conversion 邊界，沒有重構 converter 或引入 Metal kernel。
upstream main 的 software path 同樣使用 pinned/private stream，不能直接作 MPS path。

現有 callers 的直接 CUDA worker／memory 呼叫屬 #6；software encoder 屬 #11，
CLI integration 屬 #12。此 PR 僅驗證 decoder 與跨執行緒 handoff，沒有提前實作它們。

Debug：初版在 blocking copy 後增加 `torch.mps.synchronize()`，兩個 decoder
並行時於該呼叫 native abort。移除多餘的全域 sync，使用 blocking copy 的完成邊界後，
兩個 worker 各重開 8 次 reader（272 decoded frames）的 pixel/PTS/lifetime stress 通過，
Python 3.12/3.13 均正常。沒有用 skip、鎖住整段 decode 或弱化斷言繞過並行問題。

## Tests

Repository root 執行；GPU tests 在本機可存取 MPS 的執行環境完成。

```bash
PY312=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
PY313=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-313/bin/python
EVIDENCE=/Users/kaho/jasna-mac/verification-evidence/issue10

JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE" JASNA_DECODE_BACKEND=pyav-sw "$PY312" -m pytest -q -s tests/test_mps_video_decode.py tests/test_video_decoder_backends.py tests/test_video_decoder_seek.py tests/test_video_decoder_amd_path.py tests/test_video_decoder.py tests/test_video_decoder_software.py
# 72 passed, 25 skipped；新增 focused tests 34 個全通過，CPU/MPS 無 skip。

JASNA_DECODE_BACKEND=pyav-sw "$PY313" -m pytest -q -s tests/test_mps_video_decode.py tests/test_video_decoder_backends.py tests/test_video_decoder_seek.py tests/test_video_decoder_amd_path.py tests/test_video_decoder.py tests/test_video_decoder_software.py
# 72 passed, 25 skipped；相同 focused／seek／並行驗證。

"$PY312" -m pytest -q -s tests/test_video_decoder_backends.py tests/test_video_decoder_seek.py tests/test_video_decoder_amd_path.py tests/test_video_decoder.py tests/test_video_decoder_software.py tests/test_mps_accelerator.py tests/test_mps_capabilities.py tests/test_amd_support.py tests/test_accelerator_rocm_env.py tests/test_yuv_to_rgb.py tests/test_tensor_kernel_dispatch.py tests/test_mps_tensor_fallbacks.py
# 167 passed, 26 skipped, 1 failed：既有 ROCm defaults assertion，見 Risks。

mkdir -p /tmp/jasna-issue10-rocm-base
git archive origin/feature/macos-mps jasna/__init__.py jasna/accelerator.py tests/test_accelerator_rocm_env.py | tar -x -C /tmp/jasna-issue10-rocm-base
(cd /tmp/jasna-issue10-rocm-base && "$PY312" -m pytest -q tests/test_accelerator_rocm_env.py)
# 未修改 base 同樣 1 failed, 3 passed，錯誤完全相同。

git diff --check
# PASS。
```

Focused cases 包含真實 H.264 B-frames／partial batch／RGB pitched rows／有限值、
BT.601/709 full/limited range、start offset 1.5 秒的 seek/stride、seek past EOF、
batch boundary cancel/reopen、跨執行緒 retained tensors、真正 FFmpeg H.264 invalid
packet（單一可恢復、連續 11 個清楚報錯）、真實 10-bit 及 H.264 PQ VUI rejection。
CUDA calls、private streams、pinned allocation 以 forbidden guards 保護；實際 MPS
copy 與 tensor operation 並未 mock。25 skips 為 18 個既有 CUDA software cases、
6 個 CUDA seek cases、1 個 VALI fork/hardware case；不代表 NVIDIA/AMD 實機驗證。

## M2 Pro verification

2026-10-10：Apple M2 Pro，32 GiB unified memory，macOS 27.0.1／26A434，arm64。
Python 3.12.4／3.13.13，Torch 2.12.0，PyAV 18.1.0，NumPy 2.5.3，
system FFmpeg／ffprobe 8.0.1。兩個 Python 的 MPS built／available 都為 True/True。
`PYTORCH_ENABLE_MPS_FALLBACK` 未設定。未使用 model weights，亦未修改其唯讀目錄。

```bash
"$PY312" --version
"$PY312" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(),torch.backends.mps.is_available())'
"$PY313" --version
"$PY313" -c 'import torch,av,numpy; print(torch.__version__); print(torch.backends.mps.is_built(),torch.backends.mps.is_available()); print(av.__version__,numpy.__version__); print(av.library_versions)'
"$PY312" -c 'import av,os; print(av.library_versions); print("PYTORCH_ENABLE_MPS_FALLBACK",os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize
ffmpeg -version
ffprobe -version

ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,width,height,pix_fmt,nb_frames,nb_read_frames,duration,avg_frame_rate,time_base,start_pts -of json "$EVIDENCE/decode-roundtrip-mps.mp4"
ffmpeg -v error -i "$EVIDENCE/decode-roundtrip-mps.mp4" -f null -
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,width,height,nb_read_frames,duration -of json "$EVIDENCE/decode-roundtrip-cpu.mp4"
ffmpeg -v error -i "$EVIDENCE/decode-roundtrip-cpu.mp4" -f null -
# 兩者 H.264 1920x1080、30 frames、1 秒；MPS output yuv420p、30 fps、start_pts=0。
# PyAV／FFmpeg 均完整 decode 成功，PyAV 逐幀驗證輸出 timestamp。
```

PyAV bundled libavcodec=62.28.102、libavformat=62.12.102、libswscale=9.5.102；
與 system FFmpeg 8.0.1 的 libraries 分開記錄。

## Expected / observed result

- [x] 短 H.264 8-bit：真正 MPS uint8 contiguous BCHW，RGB／PTS 與 direct PyAV reference 完全一致。
- [x] seek／EOF／corrupt packets／cancel：正確完成或可預期 VideoDecodeError，沒有 CUDA streams。
- [x] CPU↔MPS lifetime：blocking copy、unpinned host batch、retained tensor／雙 reader stress 通過。
- [x] decoder capability tests：CPU/MPS 實機通過，NVIDIA/AMD structural regression 通過。

1080p H.264 最終觀察：Python 3.12 MPS decode/upload 30 frames 用 0.083 秒
（約 361.5 fps），Python 3.13 0.084 秒（約 359 fps）；包括 decode、RGB conversion、
CPU staging 及 blocking upload，不包括 reference／encode，也不是端到端 benchmark。
既有 1080p HEVC seek fixture 同時通過；Python 3.12 MPS sequential 約 551 fps、
seek 約 408 fps。這些短片數值會受 warm cache、batch size 及主機負載影響。

## Risks

較廣 regression 的唯一失敗：`test_rocm_pins_the_allocator_and_miopen_find_mode`，
base 的 `apply_rocm_env_defaults` 額外設定 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`，
test 的 exact dictionary 尚未包含該鍵。已在未修改 base 重現，沒有擴大 Issue #10 修改。
Full suite 未執行；本機無 NVIDIA/AMD hardware，不能宣稱其實機全部通過。

可見既有 `torch.jit.interface` deprecation，以及 PyAV/OpenCV 同時載入不同
libavdevice 的 duplicate Objective-C class warnings。本次 decode／roundtrip 均成功。
長片、HDR、10-bit、VideoToolbox、完整 pipeline 及長時間記憶體峰值未獲此驗證。
沒有阻止 Issue #10 交付的 prerequisite／weights blocker。

## Unsupported / fallback behavior

CPU fallback 僅為 FFmpeg software decode 及 RGB24 color conversion／host staging；
輸出明確 copy 到 MPS，沒有整個 model/pipeline 移至 CPU、沒有把 MPS 當 CUDA。
Apple `vali`／`pyav-hw` 延續既有明確拒絕。>8-bit（包括 P010）、PQ／HLG 不支援，
不作隱含 SDR tone mapping。影片缺失或錯誤的 HDR metadata 無法保證被辨識。

測試中的 PyAV libx264 僅輸出無音訊診斷 roundtrip；尚未接通 Jasna encoder／音訊 mux，
也未執行 detection/restoration。這不是 #11/#12 的端到端交付。

## Issue

Closes #10
