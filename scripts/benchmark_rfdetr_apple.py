"""Real-checkpoint parity and completed detector latency, single-thread only.

No quality/resolution/threshold changes. Reports cold and warm public calls,
raw parity, selected parity, framework handoffs and memory. E2E remains separate.
"""

from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch
import av
import psutil
import torch

from jasna.mosaic.detection_registry import build_detection_model


def timing(fn, runs=5):
    times = []
    for _ in range(runs + 1):
        torch.mps.synchronize()  # Isolated benchmark only; never pipeline instrumentation.
        start = time.perf_counter()
        result = fn()
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    return result, {
        "cold_seconds": times[0],
        "warm_seconds": times[1:],
        "median_seconds": statistics.median(times[1:]),
    }


def raw_metrics(actual, reference):
    result = {}
    for key, ref in reference.items():
        actual_tensor = actual[key].cpu()
        delta = actual_tensor - ref.cpu()
        result[key] = {
            "shape": list(actual_tensor.shape),
            "finite": bool(actual_tensor.isfinite().all()),
            "max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
        }
    return result


def selected_metrics(actual, reference):
    counts = [len(b) for b in actual.boxes_xyxy]
    refs = [len(b) for b in reference.boxes_xyxy]
    box_error, ious = [], []
    for box, refbox, mask, refmask in zip(
        actual.boxes_xyxy, reference.boxes_xyxy, actual.masks, reference.masks
    ):
        if box.shape != refbox.shape:
            continue
        if len(box):
            box_error.append(float(abs(box - refbox).max()))
            a, r = mask.cpu(), refmask.cpu()
            ious.extend(
                ((a & r).sum((1, 2)) / (a | r).sum((1, 2)).clamp_min(1)).tolist()
            )
    return {
        "counts": counts,
        "reference_counts": refs,
        "positive_decisions_match": [n > 0 for n in counts] == [n > 0 for n in refs],
        "counts_match": counts == refs,
        "max_box_pixels": max(box_error, default=0.0),
        "min_mask_iou": min(ious, default=1.0),
    }


def read_frames(path, indexes):
    result = []
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in indexes:
                result.append(
                    torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                )
            if len(result) == len(indexes):
                break
    if len(result) != len(indexes):
        raise ValueError("Source does not contain all requested frame indices")
    return torch.stack(result).to("mps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs=4, default=[0, 120, 121, 122])
    parser.add_argument(
        "--backends", nargs="+", choices=["mlx", "coreml"], default=["mlx", "coreml"]
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    assert (
        torch.backends.mps.is_available()
        and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"
    )
    frames = read_frames(args.input, args.frames)
    report = {
        "input": str(args.input),
        "frames": args.frames,
        "threshold": 0.35,
        "resolution": 576,
        "rows": [],
        "versions": {
            p: importlib.metadata.version(p) for p in ["torch", "rfdetr", "numpy"]
        },
    }
    with args.weights.open("rb") as handle:
        report["checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    with torch.inference_mode():
        os.environ["JASNA_APPLE_RFDETR_BACKEND"] = "torch"
        os.environ["JASNA_MPS_RFDETR_EXPORT"] = "0"
        ref = build_detection_model(
            "rfdetr-v6",
            args.weights,
            batch_size=4,
            device=torch.device("mps"),
            score_threshold=0.35,
            fp16=False,
        )
        for batch in (1, 2, 4):
            x, preprocess = timing(lambda: ref._preprocess(frames[:batch]), args.runs)
            reference, forward = timing(lambda: ref._infer(x), args.runs)
            selected_ref, public_ref = timing(
                lambda: ref(frames[:batch], target_hw=frames.shape[-2:]), args.runs
            )
            _, torch_postprocess = timing(
                lambda: ref._postprocess(
                    pred_boxes=reference["dets"],
                    pred_logits=reference["labels"],
                    pred_masks=reference["masks"],
                    target_hw=frames.shape[-2:],
                    score_threshold=0.35,
                    max_select=16,
                ),
                args.runs,
            )
            ref_cpu = {k: v.cpu() for k, v in reference.items()}
            for backend in args.backends:
                os.environ["JASNA_APPLE_RFDETR_BACKEND"] = backend
                start = time.perf_counter()
                model = build_detection_model(
                    "rfdetr-v6",
                    args.weights,
                    batch_size=4,
                    device=torch.device("mps"),
                    score_threshold=0.35,
                    fp16=False,
                )
                row = {
                    "batch": batch,
                    "backend": backend,
                    "load_seconds": time.perf_counter() - start,
                    "torch_preprocess": preprocess,
                    "torch_forward": forward,
                    "torch_public": public_ref,
                    "torch_postprocess_and_selected_handoff": torch_postprocess,
                }
                raw, row["raw_with_handoffs"] = timing(
                    lambda: model._infer(x), args.runs
                )
                row["raw"] = raw_metrics(raw, ref_cpu)
                repeated = model._infer(x)
                row["repeat_exact"] = all(
                    torch.equal(raw[k].cpu(), repeated[k].cpu()) for k in raw
                )
                actual, row["public"] = timing(
                    lambda: model(frames[:batch], target_hw=frames.shape[-2:]),
                    args.runs,
                )
                row["selected"] = selected_metrics(actual, selected_ref)
                # Compare selected score vectors with the same top-16 / strict > threshold policy.
                ap = raw["labels"].sigmoid().flatten(1).topk(16, dim=1).values.cpu()
                rp = ref_cpu["labels"].sigmoid().flatten(1).topk(16, dim=1).values
                row["selected"]["max_score_error"] = float(abs(ap - rp).max())
                _, row["input_download_completed_gpu"] = timing(
                    lambda: x.cpu(), args.runs
                )
                # Cache completed native outputs to isolate postprocess/output
                # handoff from model forward and input download. No production patch.
                cached = model.runner._raw(x)
                if backend == "mlx":
                    model.runner.mx.eval(cached)
                with patch.object(model.runner, "_raw", return_value=cached):
                    _, row["postprocess_and_selected_handoff"] = timing(
                        lambda: model.runner.detect(
                            x,
                            target_hw=frames.shape[-2:],
                            score_threshold=0.35,
                            max_select=16,
                        ),
                        args.runs,
                    )
                if backend == "mlx":
                    mx = model.runner.mx
                    host = x.cpu().numpy()

                    def upload():
                        value = mx.array(host)
                        mx.eval(value)
                        return value

                    z, row["mlx_input_copy"] = timing(upload, args.runs)

                    def native():
                        value = model.runner._forward(z)
                        mx.eval(value)
                        return value

                    _, row["native_forward"] = timing(native, args.runs)
                    row["mlx_memory"] = {
                        "active": mx.get_active_memory(),
                        "peak": mx.get_peak_memory(),
                        "cache": mx.get_cache_memory(),
                    }
                else:
                    native_model = model.runner._models[batch]
                    input_name = model.runner.manifest["models"][str(batch)]["input"]
                    host = x.cpu().numpy()
                    _, row["native_forward"] = timing(
                        lambda: native_model.predict({input_name: host}),
                        args.runs,
                    )
                row["memory"] = {
                    "rss": psutil.Process().memory_info().rss,
                    "mps_allocated": torch.mps.current_allocated_memory(),
                    "mps_driver": torch.mps.driver_allocated_memory(),
                    "swap": psutil.swap_memory().used,
                }
                row["speedup"] = (
                    public_ref["median_seconds"] / row["public"]["median_seconds"]
                )
                s = row["selected"]
                row["correctness_pass"] = (
                    all(v["finite"] for v in row["raw"].values())
                    and row["repeat_exact"]
                    and s["counts_match"]
                    and s["positive_decisions_match"]
                    and s["max_box_pixels"] <= 1
                    and s["min_mask_iou"] >= 0.995
                    and s["max_score_error"] <= 0.001
                )
                report["rows"].append(row)
                save()
                print(json.dumps(row), flush=True)
                model.close()
                model = None
                del raw, repeated, actual, cached
        ref.close()
    with args.weights.open("rb") as handle:
        assert (
            hashlib.file_digest(handle, "sha256").hexdigest()
            == report["checkpoint_sha256"]
        )
    report["correctness_pass"] = all(row["correctness_pass"] for row in report["rows"])
    save()


if __name__ == "__main__":
    main()
