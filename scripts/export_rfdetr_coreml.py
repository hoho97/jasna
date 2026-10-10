"""Export v6 with an isolated, pinned official RF-DETR checkout; never replace rfdetr.

Run in a fresh process with jasna[macos,macos-coreml]. --official-source is a
checkout of the audited upstream commit. Artifacts are explicit, not auto-downloaded.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OFFICIAL_REVISION = "eca736acab9f6fe93f2cbc82a1d9f6edfb53f0ea"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if "rfdetr" in sys.modules:
        raise RuntimeError("Exporter must run in a fresh process")
    revision = subprocess.check_output(
        ["git", "-C", str(args.official_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != OFFICIAL_REVISION:
        raise ValueError(
            f"Expected audited official RF-DETR {OFFICIAL_REVISION}, got {revision}"
        )
    if args.output_dir.resolve().is_relative_to(args.weights.resolve().parent):
        raise ValueError(
            "Export directory must be outside the readonly weights directory"
        )
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing artifact manifest")
    sys.path.insert(0, str(args.official_source / "src"))
    import torch
    import rfdetr
    import coremltools as ct
    from jasna.mosaic.mlx_rfdetr.checkpoint import mapped_tensors

    state = torch.load(args.weights, map_location="cpu", weights_only=False)["model"]
    mapped_tensors(state)  # Reject nonempty keypoint metadata.
    with args.weights.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "checkpoint_sha256": digest,
        "official_revision": revision,
        "torch": torch.__version__,
        "coremltools": ct.__version__,
        "contract": {
            "resolution": 576,
            "queries": 200,
            "classes": 3,
            "mask_hw": [144, 144],
            "precision": "float32",
        },
        "models": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for batch in (1, 2, 4):
        wrapper = rfdetr.RFDETRSegMedium(
            num_classes=2,
            resolution=576,
            pretrain_weights=str(args.weights),
            device="cpu",
        )
        core = wrapper.model.model.cpu().eval()
        from rfdetr.models.weights import interpolate_position_embeddings

        strict_state = dict(state)
        interpolate_position_embeddings(strict_state, 48)
        core.load_state_dict(strict_state, strict=True)
        path = wrapper.export(
            output_dir=str(args.output_dir / f"batch{batch}"),
            format="coreml",
            batch_size=batch,
            shape=(576, 576),
            coreml_precision="float32",
        )
        model = ct.models.MLModel(str(path), skip_model_load=True)
        spec = model.get_spec()
        # The official export contract is boxes, logits, masks in that order.
        manifest["models"][str(batch)] = {
            "path": str(Path(path).relative_to(args.output_dir)),
            "input": spec.description.input[0].name,
            "outputs": dict(
                zip(
                    ("dets", "labels", "masks"),
                    (o.name for o in spec.description.output),
                    strict=True,
                )
            ),
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    with args.weights.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == digest


if __name__ == "__main__":
    main()
