# Issue #4：GPU preflight 與 backend 功能門檻

驗證日期：2026-10-10（香港）。範圍僅限 #4，base 為 `e5cc011`；Epic #1 指定的 prerequisite #3 已透過 PR #20 合併，#5、#7 亦已存在。Issue #4／Epic #1 當時均無 comments。

## 架構審查

| Caller → callee | 原有問題／採用處理 |
|---|---|
| console script、`python -m jasna` → `main` → `_check_system` → `os_utils` | MPS 先檢查 built／available，使用 macOS 診斷；不呼叫 CUDA capability、driver 或 `nvidia-smi`。CUDA 預設裝置與 NVIDIA driver 下限保持原樣。 |
| `main` → benchmark、streaming、image restore、LTX download、session factory | 在執行／載入專用模組前，以共用 capabilities 拒絕未支援選項。secondary 預設仍為 `none`。 |
| GUI／CLI → session factory → secondary、`_ltx_model_files`、`provide_restoration_models` | 共用 gate 保護直接呼叫及延後載入的 segment 模型；RTX／UNet 不先 import SDK／protection。AMD 既有 secondary 限制保留。 |
| session factory → engine compiler → detection registry／TensorRT subprocess | #5 已讓 Apple `.pt` checkpoint 不探查 engine；本次加上 parent／child compilation 的非 NVIDIA 停止門檻，直接 child 請求 UNet 亦明確拒絕。 |
| GUI engine preflight → engine paths | 新增明確 `device` 參數；MPS 不建立／探查 NVIDIA cache。GUI 預設裝置選擇仍屬 #16，沒有聲稱 GUI 已支援 MPS。 |
| VideoReader → VALI／PyAV／NVDEC／AMF | 原本已按 vendor 選擇 NVIDIA／AMD 實作；額外拒絕 Apple 明確指定 `vali`／`pyav-hw`，不默默降為 software。 |
| CLI encoder settings → VideoEncoder | 既有 VideoEncoder 已拒絕非 NVIDIA／AMD；CLI 提早給出 #11 software encoder 尚未實作的訊息，避免誤用 NVENC／AMF。 |

已讀取相關 accelerator、engine paths／compiler、session config／factory、CLI／module entry point、GUI engine preflight、RTX／UNet secondary、TRT package、RF-DETR Torch／TRT runner、BasicVSR++ 載入、video decoder／encoder 與相關測試，並對照 `upstream/main` 的 session factory／AMD 路徑。沒有修改 upstream、實作 #8–#12 或完成 #18 的可選功能。

Capabilities 增加 TensorRT、secondary、LTX、advanced video 支援資訊；NVIDIA 與 ROCm 保留原有 execution primitives。TVAI 使用外部 Topaz FFmpeg，拒絕理由是此 backend 未驗證，並非宣稱 TVAI 是 NVIDIA-only。

## 重現命令與結果

以下命令皆於 `/Users/kaho/jasna-mac/jasna` 執行。測試使用既有 Python 3.12 環境，從目前 checkout 匯入 source；console script 子程序明確設定目前 checkout 的 `PYTHONPATH`。

```sh
/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python -m pytest -q tests/test_main_validation.py tests/test_engine_compiler.py tests/test_restorer_lazy_import.py
# 49 passed
/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python -m pytest -q tests/test_mps_capabilities.py
# 34 passed
/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python -m pytest -q tests/test_main.py tests/test_main_cli_device.py tests/test_session_factory.py tests/test_benchmark.py tests/test_os_utils.py tests/test_amd_support.py tests/test_gui_engine_preflight.py tests/test_mps_accelerator.py tests/test_mps_model_loading.py tests/test_ltx_model_files.py tests/test_ltx_trial.py tests/test_video_decoder_backends.py tests/test_video_decoder_amd_path.py tests/test_video_encoder_unit.py
# 354 passed, 14 skipped
```

合計 **437 passed、14 skipped、0 failed**。Skipped 是 Windows-only、CUDA／TensorRT 實機、未設定 real-weights opt-in，以及缺少 LTX raw weights 的 cases；不能解讀成已實測 NVIDIA／AMD 硬體。NVIDIA／AMD 邏輯在 M2 Pro 以原有 mocks 驗證。

原有 CLI／composition／encoder unit tests 會在模擬 CUDA 流程時讀取 host 的 Torch build。新增明確 opt-in fixtures 模擬 NVIDIA build／SDK constructors；三個已 mock converter／DLPack 的 encoder tests 使用預先配置 CPU 測試 buffer。原有 stream、同步、順序與輸出 assertions 全數保留。未全域啟用 mock 或跳過不通過的 assertion。

## M2 Pro verification

```sh
/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python --version
/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize
```

實際結果：Apple M2 Pro、34,359,738,368 bytes（32 GiB）、arm64、macOS 27.0.1（26A434）、Python 3.12.4、Torch 2.12.0、MPS built／available = true／true。沙盒中 availability=false；真實 GPU 驗證在主機執行。

Dependency versions：torchvision 0.27.0、rfdetr 1.8.3、transformers 5.1.0、PyAV 18.1.0、numpy 2.5.3、mmengine 0.10.7、pytest 9.1.1、FFmpeg／ffprobe 8.0.1。

```sh
ffmpeg -v error -f lavfi -i 'testsrc2=size=64x64:rate=4' -frames:v 4 -c:v libx264 -pix_fmt yuv420p -y /tmp/issue4-preflight.mp4
ffprobe -v error -count_frames -show_entries stream=codec_name,width,height,nb_read_frames -show_entries format=duration -of json /tmp/issue4-preflight.mp4
ffmpeg -v error -i /tmp/issue4-preflight.mp4 -f null -
/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python -m scripts.verify_mps_preflight --weights-dir '/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights' --video /tmp/issue4-preflight.mp4
```

全部 exit 0。ffprobe：H.264、64×64、4 frames、1 秒；完整 decode 無錯誤。PyAV 實際 software decode 4 frames 並 upload 至 MPS。這是獨立 media smoke，並非 Jasna restoration video export。

驗證程式以 import guard 禁止實際載入 `tensorrt`、`torch_tensorrt`、`tensorrt_libs`、`nvvfx`、`python_vali`、`jasna.protection`；允許 Torch 的 optional-package `find_spec` 探查。結果：

- 真正 MPS FP32 convolution：shape `[2,8,32,32]`、finite，與 CPU reference 接近（rtol=1e-4／atol=1e-5）。單次 operation 約 0.0414 秒，包含 transfer 與首次 dispatch，不是穩定效能 benchmark。
- BasicVSR++ 真實 checkpoint 的 `feat_extract`：`[1,64,16,16]`、FP32、MPS、finite。
- RF-DETR v6 真實 checkpoint 的 `class_embed`：`[1,3]`、FP32、MPS、finite。
- 兩個免費模型的 parameters／buffers 均位於 MPS，浮點 tensors 為 FP32 且 finite。
- `_check_system` 通過；MPS checkpoint compilation request 不產生 TensorRT engine。
- 已安裝 `jasna` console script 與 `python -m jasna` 均以 exit 2 在執行前具體拒絕 RTX secondary。
- 真實 MPS VideoReader 對強制 VALI／hardware decode 在 open 前拒絕。
- optional NVIDIA／protection imports = none；`PYTORCH_ENABLE_MPS_FALLBACK` 未設定。

以上只驗證 preflight／checkpoint layers；完整 RF-DETR frame inference 屬 #8、BasicVSR++ temporal inference 屬 #9。

唯讀使用的原始 weights：

| 檔案 | SHA-256 |
|---|---|
| `rfdetr-v6.pt` | `f10bedc4d105c2721e4259b8680203d51f344f73e55e85710d915619f5731b55` |
| `lada_mosaic_restoration_model_generic_v1.2.pth` | `d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee` |

## 驗收與限制

- [x] MPS CLI preflight 不受 CUDA driver／compute capability 阻擋。
- [x] 免費模型載入及真實 checkpoint layer 操作不依賴 NVIDIA／protection imports。
- [x] 未支援選項在執行前明確拒絕；secondary 預設 `none`。
- [x] NVIDIA／AMD preflight 與 engine compilation 邏輯 regression tests 通過。

#4 結果為 **PASS**。沒有本 Issue 的 blocker。完整 MPS video CLI 仍需要 #6、#8–#12；預設 CLI device 仍是 `cuda:0`，本次 MPS 需明確 `--device mps`。LTX、streaming、explicit VR、smart render、benchmark、secondary／supporter image paths、hardware decode 未開放；software encoder 未完成，CLI 明確指出 #11。沒有完成下一個 Issue。

沒有新增 runtime CPU fallback。原有 checkpoint CPU deserialization、測試 CPU reference、standalone PyAV host decode 是明確邊界，沒有把整個模型移往 CPU。

Warnings：PyAV／OpenCV 各自載入 libavdevice 產生 duplicate Objective-C class 訊息；Torch JIT／rfdetr deprecation warnings；RF-DETR positional encoding／patch size 及缺省 num_queries 推導提示。這些未導致本次測試失敗，但完整 inference／影片流程仍需後續獨立驗證。未修改第三方 binaries 以壓掉警告。
