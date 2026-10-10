"""Opt-in Jasna v6 MLX FP32 inference with owned, blocking framework handoffs."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from jasna.model_weights import load_rfdetr_checkpoint
from jasna.mosaic.detections import Detections
from jasna.mosaic.rfdetr_torch_runner import TorchTensorInfo


class RfDetrMlxRunner:
    native_postprocess = True

    def __init__(
        self,
        weights_path: Path,
        input_shapes,
        device: torch.device,
        *,
        fp16: bool,
        resolution: int,
        variant: str,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "mps" or resolution != 576 or variant != "medium":
            raise ValueError("MLX backend supports only Jasna v6 (MPS, medium, 576)")
        try:
            import mlx.core as mx
            from mlx.utils import tree_flatten
        except ImportError as exc:
            raise RuntimeError("MLX RF-DETR requires jasna[macos-mlx]") from exc
        from jasna.mosaic.mlx_rfdetr.config import JasnaV6MLXConfig
        from jasna.mosaic.mlx_rfdetr.model import RFDETRForInference
        from jasna.mosaic.mlx_rfdetr.checkpoint import mapped_tensors, validate_mapping

        self.mx = mx
        self.config = JasnaV6MLXConfig
        state = load_rfdetr_checkpoint(weights_path)["model"]
        mapped = mapped_tensors(state)
        model = RFDETRForInference(self.config)
        model.kp_active_mask = mx.zeros((0, 0))
        self.mapping_report = validate_mapping(
            mapped, dict(tree_flatten(model.parameters()))
        )
        model.load_weights(
            [(k, mx.array(v.numpy())) for k, v in mapped.items()], strict=True
        )
        # The existing rfdetr CPU-first loader antialiases 36x36 -> 48x48
        # positional embeddings when resolution is overridden from 432 to 576.
        # Reproduce that exact load-time transform, retaining original parameters
        # for strict checkpoint validation. No inference resolution is changed.
        position = state["backbone.0.encoder.encoder.embeddings.position_embeddings"]
        patch = position[:, 1:].reshape(1, 36, 36, 384).permute(0, 3, 1, 2)
        patch = F.interpolate(
            patch, size=(48, 48), mode="bicubic", align_corners=False, antialias=True
        )
        patch = patch.permute(0, 2, 3, 1).reshape(1, 2304, 384)
        fixed_position = torch.cat([position[:, :1], patch], dim=1)
        model.backbone[0].encoder.encoder.embeddings._fixed_position = mx.array(
            fixed_position.numpy()
        )
        mx.eval(model.parameters())
        self._model = model
        # MLX 0.31.2 compiled Module cache destroys Python dicts without the
        # GIL during worker TLS teardown (native dict_dealloc crash). Keep the
        # fused-SDPA eager graph; do not change Jasna's execution locks (#37).
        self._forward = model
        batch = (
            int(input_shapes[0][0])
            if not isinstance(input_shapes, dict)
            else int(next(iter(input_shapes.values()))[0])
        )
        self.input_names = ["input"]
        self.input_dtypes = {"input": torch.float32}
        self.output_names = ["dets", "labels", "masks"]
        self.outputs = {
            "dets": TorchTensorInfo((batch, 200, 4), torch.float32),
            "labels": TorchTensorInfo((batch, 200, 3), torch.float32),
            "masks": TorchTensorInfo((batch, 200, 144, 144), torch.float32),
        }

    def _raw(self, x):
        if self._forward is None:
            raise RuntimeError("MLX RF-DETR runner is closed")
        if (
            x.ndim != 4
            or tuple(x.shape[1:]) != (3, 576, 576)
            or not 1 <= x.shape[0] <= 4
        ):
            raise ValueError("MLX v6 input must be Bx3x576x576, batch 1..4")
        # MLX 0.31.2 has no Metal DLPack import. A blocking CPU snapshot completes
        # the producer's work and owns its lifetime; no extra global synchronize.
        host = x.detach().to(device="cpu", dtype=torch.float32, non_blocking=False)
        return self._forward(self.mx.array(host.numpy()))

    def infer(self, inputs):
        out = self._raw(inputs["input"])
        self.mx.eval(out)
        return {
            dst: torch.from_numpy(np.array(out[src], copy=True)).to(
                self.device, non_blocking=False
            )
            for dst, src in [
                ("dets", "pred_boxes"),
                ("labels", "pred_logits"),
                ("masks", "pred_masks"),
            ]
        }

    def detect(self, x, *, target_hw, score_threshold, max_select):
        mx = self.mx
        out = self._raw(x)
        prob = mx.sigmoid(out["pred_logits"])
        b, q, c = prob.shape
        flat = prob.reshape(b, -1)
        indices = mx.argsort(-flat, axis=1)[:, : min(max_select, q)]
        scores = mx.take_along_axis(flat, indices, axis=1)
        boxes, masks = [], []
        th, tw = target_hw
        for i in range(b):
            query = indices[i] // c
            box = mx.take(out["pred_boxes"][i], query, axis=0)
            center, size = box[:, :2], box[:, 2:]
            boxes.append(
                mx.concatenate([center - size * 0.5, center + size * 0.5], axis=-1)
                * mx.array([tw, th, tw, th])
            )
            masks.append(mx.take(out["pred_masks"][i], query, axis=0) > 0)
        mx.eval(scores, boxes, masks)
        valid = np.array(scores, copy=True) > score_threshold
        # Only selected low-resolution bool masks cross back, never Q full FP32 masks.
        return Detections(
            boxes_xyxy=[
                np.array(box, copy=True)[valid[i]] for i, box in enumerate(boxes)
            ],
            masks=[
                torch.from_numpy(np.array(mask, copy=True)[valid[i]]).to(
                    self.device, non_blocking=False
                )
                for i, mask in enumerate(masks)
            ],
        )

    def close(self):
        self._forward = None
        self._model = None
