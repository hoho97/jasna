# BasicVSR++ MPS restoration（Issue #9）

## 架構與範圍

基準 `origin/feature/macos-mps`：`8cdfae0`。#2、#3、#5、#7 已合併；
origin 為 hoho97/jasna，upstream Kruk2/jasna 僅供唯讀參考。
Issue #9、Epic #1 均無 comments。只處理 #9，沒有實作 #6／#10／#11／#12。

實際 caller/callee：`session_factory._build_basicvsrpp_pipeline` →
`BasicvsrppMosaicRestorer` → `load_model` → CPU 嚴格載入 .pth tensor state_dict → MPS。
`RestorationPipeline.prepare_and_run_primary` → `prepare_crops_for_restoration`
（resize／reflect padding／FP32）→ `raw_process`（stack／除以255）→
`BaseEditModel.forward(mode='tensor')` → `RealBasicVSR.forward_tensor` → `generator_ema`。
SPyNet 計算雙向光流，四條 propagation branches 使用 `flow_warp`／
`SecondOrderDeformableAlignment.forward` → `torchvision.ops.deform_conv2d`。
基類 `deformconv.py.forward=pass` 不在此 execution path。

`model_utils.get_module_device` 非 CUDA 回 CPU，但全 repo 無 caller；
保留為潛在相容性 inventory，沒有為未使用的 helper 擴大本次修改。
AMD 重用 Torch model，NVIDIA 保留 TensorRT split-forward。兩者的
FP16 request／dtype／routing 保持原狀；沒有修改 Windows/Linux guards。

新增 MPS FP32 policy，覆蓋 public restorer 及直接 loader：即使傳入 `fp16=True`，
仍使用 FP32 並記錄 warning。BF16 沒有開放介面；FP16／BF16 必須另行驗證才開啟。
Torch 2.12.0／torchvision 0.27.0 的 production deform／grid_sample 已通過 native MPS，
所以沒有加入 CPU op fallback、沒有開啟 `PYTORCH_ENABLE_MPS_FALLBACK`，沒有移整模型 CPU。
其他版本若缺 op，保留明確原生錯誤；不吞掉 dtype／memory／backend failures。

## 預先記錄的驗收門檻與結果

完整 restoration CPU FP32 parity：**atol=1e-3、rtol=1e-3**（nominal [0,1] scale），
在第一次 CPU 比對前記錄。獨立 production deform／flow warp probe 使用
atol=rtol=1e-4。未因結果放寬容差。

真實 checkpoint：`lada_mosaic_restoration_model_generic_v1.2.pth`（812 tensors），
模型目錄唯讀：`/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights`。
驗證前後 SHA256 相同：
`d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee`。

PyAV decode fixture `assets/test_clip1_1080p.mp4` 的 frames 120–122，
取不同高度的實際 RGB crops；經 production resize／reflect padding 變為 256²。
T=2 覆蓋光流／雙向 propagation；T=3 額外覆蓋 second-order feature／flow composition。
驗證每條 branch 實際執行 alignment，x／offset／mask／weight／bias 均留在 MPS，
feature x=(1,128,64,64)、offset=(1,288,64,64)、mask=(1,144,64,64)，
deform_groups=16、stride=1、padding=1、dilation=1。

| Python 3.12.4 | T=2 | T=3 |
|---|---:|---:|
| CPU max absolute error | 5.96e-7 | 7.75e-7 |
| CPU RMSE | 3.6e-8 | 3.9e-8 |
| mean absolute change from input | 0.004760 | 0.005025 |
| synchronized MPS primary time | 0.265 s | 0.234 s |
| CPU primary time | 0.523 s | 0.916 s |
| post-run MPS tensor allocation | 82.7 MB | 85.2 MB |
| post-run driver allocation | 215 MB | 1272 MB |

輸出 `(T,3,256,256)`、FP32、MPS、finite；有實際 restoration，非 passthrough。
raw 模型 residual 輸出未 clamp，範圍約 -0.00212 至 0.90776；CPU 也有相同 overshoot。
既有 `_run_secondary` 的 clamp／round／uint8 邊界保留，輸出 RGB uint8 正常。
時間包含 crop preparation／primary inference，不含模型載入、CPU reference、decode／encode；
只是短 crop observation，並非完整影片 FPS benchmark。記憶體為事後 snapshot，並非 peak。

## 真實 M2 Pro 環境

2026-10-10：Apple M2 Pro，32 GiB unified memory，arm64，macOS 27.0.1 / 26A434。
Python 3.12.4／3.13.13，Torch 2.12.0，torchvision 0.27.0，mmengine 0.10.7，
numpy 2.5.3，PyAV 18.1.0，FFmpeg／ffprobe 8.0.1。
兩環境主機 MPS `is_built()/is_available()` = `True/True`。
GPU 驗證於本機 sandbox 外執行；未以 sandbox 內 GPU 不可用當作 skip。

## 可重現命令與結果

於 repository root 執行：

```bash
PY312=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-312/bin/python
PY313=/Users/kaho/jasna-mac/verification-pr19/.venv-macos-313/bin/python
WEIGHTS='/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights'
EVIDENCE=/Users/kaho/jasna-mac/verification-evidence/issue9

"$PY312" --version
"$PY313" --version
"$PY312" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
"$PY313" -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(),torch.backends.mps.is_available())'
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string hw.memsize
"$PY312" -c 'import importlib.metadata as m; print({p:m.version(p) for p in ("torchvision","mmengine","numpy","av")})'
"$PY313" -c 'import importlib.metadata as m; print({p:m.version(p) for p in ("torchvision","mmengine","numpy","av")})'
ffmpeg -version
ffprobe -version

"$PY312" -m pytest -q tests/test_basicvsrpp_mosaic_restorer.py tests/test_restoration_pipeline.py tests/test_crop_buffer.py
# 最終 focused tests：45 passed, 1 skipped。

"$PY312" -m pytest -q tests/test_basicvsrpp_mosaic_restorer.py tests/test_restoration_pipeline.py tests/test_crop_buffer.py tests/test_amd_support.py tests/test_mps_model_loading.py tests/test_session_factory.py tests/test_engine_compiler.py tests/test_restorer_lazy_import.py
"$PY313" -m pytest -q tests/test_basicvsrpp_mosaic_restorer.py tests/test_restoration_pipeline.py tests/test_crop_buffer.py tests/test_amd_support.py tests/test_mps_model_loading.py tests/test_session_factory.py tests/test_engine_compiler.py tests/test_restorer_lazy_import.py
# 各 122 passed, 6 skipped：既有 fp16 CUDA/platform case + 未啟用的 4 個 opt-in weights cases。

JASNA_TEST_MODEL_WEIGHTS_DIR="$WEIGHTS" JASNA_TEST_VIDEO_OUTPUT_DIR="$EVIDENCE" "$PY312" -m pytest -q -s tests/test_mps_basicvsrpp_inference.py
JASNA_TEST_MODEL_WEIGHTS_DIR="$WEIGHTS" "$PY313" -m pytest -q -s tests/test_mps_basicvsrpp_inference.py
# 各 6 passed，沒有 skip；兩幀、三幀、直接 loader、production deform、兩種 flow padding。

ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,width,height,nb_frames,nb_read_frames,duration,avg_frame_rate -of json "$EVIDENCE/basicvsrpp-t2.mp4"
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,width,height,nb_frames,nb_read_frames,duration,avg_frame_rate -of json "$EVIDENCE/basicvsrpp-t3.mp4"
ffmpeg -v error -i "$EVIDENCE/basicvsrpp-t2.mp4" -f null -
ffmpeg -v error -i "$EVIDENCE/basicvsrpp-t3.mp4" -f null -
# H.264、256x256、30 fps；2/3 frames，0.066667/0.100000 秒；完整 decode 成功。
# PyAV 也重新 decode 全部輸出，確認 frame count／尺寸／monotonic PTS。

shasum -a 256 "$WEIGHTS/lada_mosaic_restoration_model_generic_v1.2.pth"
git diff --check
```

診斷影片只輸出 restored crops，PyAV libx264 是 CPU encode 邊界；不含音訊。
沒有宣稱完成全幅 blending／CLI／threaded video I/O；這些屬後續 issues。

## Regression、warnings 與限制

新增 tests 保護 NVIDIA split-forward 的 model/dtype/FP16 傳遞及 close、
AMD Torch routing、MPS TensorRT bypass；既有 AMD/compiler/session/pipeline tests 通過。
本機無 NVIDIA／AMD hardware，只能報告其 mock contract regression 通過。
Full suite 未執行，TensorRT SDK tests 不能於 M2 Pro 執行。

有既有 `torch.jit.interface` deprecation warning，及 PyAV/OpenCV 同時載入不同
libavdevice 的重複 Objective-C class warning；這次 decode／inference／encode 均成功。
未更動這些外部 dependencies。長 clips、其他 weights／versions、FP16／BF16 品質及
完整端到端記憶體峰值不在本次已驗證範圍。沒有真正 blocker。
