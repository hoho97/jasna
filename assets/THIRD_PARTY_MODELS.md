# Model licenses and provenance

Hashes below identify the v0.11.0 candidate artifacts. Release preparation
must fail or update this file when a bundled model changes.

## Lada BasicVSR++ mosaic restoration

- Jasna model name: `basicvsrpp`
- File: `lada_mosaic_restoration_model_generic_v1.2.pth`
- Upstream: <https://huggingface.co/ladaapp/lada/blob/3bfd69ffc21518bde80ba6b61696d51efd0a398b/lada_mosaic_restoration_model_generic_v1.2.pth>
- Upstream revision: `3bfd69ffc21518bde80ba6b61696d51efd0a398b`
- SHA-256: `d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee`
- License: AGPL-3.0
- Copyright: ladaapp and contributors

This checkpoint is redistributed unmodified.

## Lada YOLO v4 fast mosaic detection

- Jasna model name: `lada-yolo-v4`
- File: `lada_mosaic_detection_model_v4_fast.pt`
- Upstream: <https://huggingface.co/ladaapp/lada/blob/404620fe2f6b72657b92f76e62af914c8b3ee686/lada_mosaic_detection_model_v4_fast.pt>
- Upstream revision: `404620fe2f6b72657b92f76e62af914c8b3ee686`
- SHA-256: `9a6b660d1d3e3797d39515e08b0e72fcc59815f38279faa7a4ab374ab2c1e3b4`
- License: AGPL-3.0
- Copyright: ladaapp and contributors

This checkpoint is redistributed unmodified.

## LTX restoration (LTX-2.5 video VAE and fine-tuned transformer)

- Jasna model name: `ltx`
- Folder: `model_weights/ltx-restore/`
- `vae.safetensors`: Lightricks `ltx-2.5-video-vae-bf16.safetensors`, redistributed unmodified
- `vae.safetensors` SHA-256: `847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3`
- `vae-decoder.safetensors`: Jasna fine-tune of the LTX-2.5 video VAE decoder
- `transformer.safetensors`: Jasna fine-tune of the LTX-2.5 22B transformer (video path only, INT8)
- Upstream: [Lightricks LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)
- License: LTX-2.x Community License (`assets/licenses/LTX-2.x-Community-License.txt`)
- Copyright: Lightricks Ltd.; fine-tunes 2026 Kruk2

These files are not part of release packages yet; hashes for the fine-tuned
files are added when they are.

## Jasna RF-DETR v6

- Jasna model name: `rfdetr-v6`
- NVIDIA file: `rfdetr-v6.onnx`
- NVIDIA SHA-256: `b6555cfce325d1d8bc413422cd46f3a453511a246c6f29ce652382998049d825`
- AMD file: `rfdetr-v6.pt`
- AMD SHA-256: `f10bedc4d105c2721e4259b8680203d51f344f73e55e85710d915619f5731b55`
- Architecture/source: [RF-DETR 1.8.3](https://github.com/roboflow/rf-detr/tree/3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86)
- License: Apache-2.0
- Copyright: 2026 Kruk2

This project-trained RF-DETR Seg Medium checkpoint uses classes
`Background` and `mosaic`. The ONNX and PyTorch files are deployment
formats of the same trained model.

## Jasna RF-DETR VR v1

- Jasna model name: `rfdetr-vr-v1`
- NVIDIA file: `rfdetr-vr-v1.onnx`
- NVIDIA SHA-256: `6e2ed2043851dccb97f21deda38dc20ea2b8e265e682359752e815c600030a40`
- AMD file: `rfdetr-vr-v1.pt`
- AMD SHA-256: `55543c83911921ef79cd8cae8540bd25e34c7daf488e77f79d233d6926973a2e`
- Architecture/source: [RF-DETR 1.8.3](https://github.com/roboflow/rf-detr/tree/3bd6bffbcb13cac3a5b1c37da5a0fd5453b50c86)
- License: Apache-2.0
- Copyright: 2026 Kruk2

This project-trained RF-DETR Seg Large checkpoint is trained for side-by-side
VR material. The ONNX and PyTorch files are deployment formats of the same
trained model.

## ZeLeFans VR Mosaic Detection v2 accurate

- Jasna model name: `zelefans-vr-yolo-v2`
- Upstream: <https://huggingface.co/zelefans/vrmr>
- Upstream project: <https://codeberg.org/zelefans/vr_remove_mosaic>
- Pinned revision: `0f65a21133335f9a4ec6fc5d7da8d3385bfdb8b1`
- Upstream file: `lada_vr_mosaic_detection_model_v2_accurate.pt`
- SHA-256: `91fe7a48b0e9edf51361918c8a30f752c64511005e643343a7382d951f3fe0f8`
- License: Apache-2.0

This optional checkpoint is not bundled in the standard v0.11.0 package.

## Supporter models

`unet-4x.onnx.enc` and the encrypted SD 1.5 Jasna checkpoint are
project-trained, proprietary supporter models. Copyright 2026 Kruk2. They are
provided only for use with Jasna by a holder of a valid supporter key.
Redistribution, extraction, modification, and use outside Jasna are not
granted.

The base v0.11.0 release may include `unet-4x.onnx.enc`; SD 1.5 is downloaded
separately when requested. The protection implementation and supporter-model
terms are separate from Jasna's AGPL-covered public application source.

## Windows 0.10.0 weights reused by the macOS source checkout

Verified read-only on Apple M2 Pro (32 GB), 2026-10-10. The six free files
match the v0.11.0 hashes above; no conversion, overwrite, or deletion is needed.
Original directory:
`/Users/kaho/jasna-mac/Windows Release/jasna-windows-0.10.0.7z/model_weights`.

| File | Bytes | SHA-256 | Format / version / use on Apple |
| --- | ---: | --- | --- |
| `lada_mosaic_restoration_model_generic_v1.2.pth` | 78441770 | `d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee` | Lada v1.2, plain state_dict, 812 FP32 tensors; BasicVSR++ restoration. No producer version metadata in the state_dict. |
| `lada_mosaic_detection_model_v4_fast.pt` | 5981796 | `9a6b660d1d3e3797d39515e08b0e72fcc59815f38279faa7a4ab374ab2c1e3b4` | YOLO v4 fast; serialized Ultralytics SegmentationModel, includes FP16 storages. Checkpoint metadata: Ultralytics 8.3.203; loaded with 8.4.174. P1 detector; loading verification does not establish full inference support. |
| `rfdetr-v6.pt` | 141702539 | `f10bedc4d105c2721e4259b8680203d51f344f73e55e85710d915619f5731b55` | RF-DETR Seg Medium v6, model + args checkpoint, 573 tensors (572 FP32, 1 bool), class_embed [3,256]; P0 detector. Loaded with rfdetr 1.8.3; no producer version recorded in checkpoint args. |
| `rfdetr-vr-v1.pt` | 145990932 | `55543c83911921ef79cd8cae8540bd25e34c7daf488e77f79d233d6926973a2e` | RF-DETR Seg Large VR v1, model + args checkpoint, 573 tensors (572 FP32, 1 bool), class_embed [2,256]. Loaded with rfdetr 1.8.3; VR inference remains separately gated. |
| `rfdetr-v6.onnx` | 146520308 | `b6555cfce325d1d8bc413422cd46f3a453511a246c6f29ce652382998049d825` | ONNX IR 8, standard opset 17, producer pytorch 2.12.0, input → dets/labels/masks. Portable graph for NVIDIA's TensorRT path; not an MPS checkpoint. |
| `rfdetr-vr-v1.onnx` | 148882087 | `6e2ed2043851dccb97f21deda38dc20ea2b8e265e682359752e815c600030a40` | ONNX IR 8, standard opset 17, producer pytorch 2.12.0, input → dets/labels/masks; not loaded on MPS. |
| `unet-4x.onnx.enc` | 68876435 | `752db7302ee936e7c825809c06539af325aed2c0a0f4d3d6d1f03bdf55048890` | Encrypted proprietary supporter model; inner format/version not inspected. Unsupported on Apple/MPS; not decrypted or loaded. |

A `.pt` suffix does not imply a state_dict: RF-DETR and YOLO have different
schemas. RF-DETR and YOLO pickle loading requires trusted checkpoints and
compatible Python classes. BasicVSR++ on MPS uses `weights_only=True` and
strict state_dict matching. All deserialize on CPU before final MPS placement.
