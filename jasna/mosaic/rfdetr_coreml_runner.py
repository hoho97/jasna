"""Opt-in, pre-exported Jasna v6 Core ML FP32 runner (CPU_AND_GPU)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch

from jasna.mosaic.detections import Detections
from jasna.mosaic.rfdetr_torch_runner import TorchTensorInfo


def validate_manifest(root: Path, weights_path: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    with weights_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if manifest.get("checkpoint_sha256") != digest:
        raise ValueError("Core ML artifact checkpoint SHA256 mismatch")
    if manifest.get("contract") != {
        "resolution": 576,
        "queries": 200,
        "classes": 3,
        "mask_hw": [144, 144],
        "precision": "float32",
    }:
        raise ValueError("Core ML artifact is not the validated Jasna v6 FP32 contract")
    if sorted(manifest.get("models", {})) != ["1", "2", "4"]:
        raise ValueError("Core ML manifest must contain static batches 1, 2, 4")
    for row in manifest["models"].values():
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_dir():
            raise ValueError("Core ML model path must exist inside artifact directory")
        if set(row["outputs"]) != {"dets", "labels", "masks"}:
            raise ValueError("Core ML manifest output names are incomplete")
    return manifest


class RfDetrCoreMLRunner:
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
            raise ValueError(
                "Core ML backend supports only Jasna v6 (MPS, medium, 576)"
            )
        try:
            import coremltools as ct
        except ImportError as exc:
            raise RuntimeError("Core ML RF-DETR requires jasna[macos-coreml]") from exc
        directory = os.environ.get("JASNA_RFDETR_COREML_DIR")
        if not directory:
            raise ValueError(
                "Set JASNA_RFDETR_COREML_DIR to an exported v6 artifact directory"
            )
        self.root = Path(directory)
        self.manifest = validate_manifest(self.root, weights_path)
        self.ct = ct
        self._models = {}
        self._closed = False
        batch = (
            int(input_shapes[0][0])
            if not isinstance(input_shapes, dict)
            else int(next(iter(input_shapes.values()))[0])
        )
        if not 1 <= batch <= 4:
            raise ValueError("Core ML v6 supports batch 1..4")
        self.input_names = ["input"]
        self.input_dtypes = {"input": torch.float32}
        self.output_names = ["dets", "labels", "masks"]
        self.outputs = {
            "dets": TorchTensorInfo((batch, 200, 4), torch.float32),
            "labels": TorchTensorInfo((batch, 200, 3), torch.float32),
            "masks": TorchTensorInfo((batch, 200, 144, 144), torch.float32),
        }

    def _raw(self, x):
        if self._closed:
            raise RuntimeError("Core ML RF-DETR runner is closed")
        b = x.shape[0]
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 576, 576) or not 1 <= b <= 4:
            raise ValueError("Core ML v6 input must be Bx3x576x576, batch 1..4")
        batch = next(n for n in (1, 2, 4) if n >= b)
        row = self.manifest["models"][str(batch)]
        if batch not in self._models:
            self._models[batch] = self.ct.models.MLModel(
                str(self.root / row["path"]),
                compute_units=self.ct.ComputeUnit.CPU_AND_GPU,
            )
        host = (
            x.detach().to(device="cpu", dtype=torch.float32, non_blocking=False).numpy()
        )
        if batch != b:
            host = np.concatenate([host, np.repeat(host[-1:], batch - b, axis=0)])
        out = self._models[batch].predict({row["input"]: host})
        result = {key: out[name][:b] for key, name in row["outputs"].items()}
        for key, shape in [
            ("dets", (b, 200, 4)),
            ("labels", (b, 200, 3)),
            ("masks", (b, 200, 144, 144)),
        ]:
            if result[key].shape != shape or result[key].dtype != np.float32:
                raise RuntimeError(
                    f"Invalid Core ML {key} contract: {result[key].shape}/{result[key].dtype}"
                )
        return result

    def infer(self, inputs):
        return {
            key: torch.from_numpy(np.array(value, copy=True)).to(
                self.device, non_blocking=False
            )
            for key, value in self._raw(inputs["input"]).items()
        }

    def detect(self, x, *, target_hw, score_threshold, max_select):
        raw = self._raw(x)
        logits = raw["labels"]
        prob = 1 / (1 + np.exp(-logits))
        b, q, c = prob.shape
        flat = prob.reshape(b, -1)
        indices = np.argsort(-flat, axis=1)[:, : min(max_select, q)]
        scores = np.take_along_axis(flat, indices, axis=1)
        boxes, masks = [], []
        th, tw = target_hw
        for i in range(b):
            query = indices[i] // c
            valid = scores[i] > score_threshold
            box = raw["dets"][i, query][valid]
            center, size = box[:, :2], box[:, 2:]
            boxes.append(
                np.concatenate([center - 0.5 * size, center + 0.5 * size], axis=-1)
                * np.array([tw, th, tw, th], dtype=np.float32)
            )
            mask = np.array(raw["masks"][i, query][valid] > 0, copy=True)
            masks.append(torch.from_numpy(mask).to(self.device, non_blocking=False))
        return Detections(boxes_xyxy=boxes, masks=masks)

    def close(self):
        self._closed = True
        self._models.clear()
