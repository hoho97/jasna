"""Apple host-frame encoding policy, independent of CUDA/AMF controls."""
import os


def apple_encode_backend() -> str:
    backend = os.environ.get("JASNA_ENCODE_BACKEND", "software")
    if backend not in {"software", "videotoolbox", "auto"}:
        raise ValueError(f"Unknown JASNA_ENCODE_BACKEND: {backend!r}")
    return backend


def validate_videotoolbox_settings(settings: dict[str, object], codec: str) -> None:
    if codec not in {"h264", "hevc"}:
        raise ValueError("VideoToolbox supports only 8-bit SDR H.264/HEVC")
    invalid = set(settings) - {"b", "g"}
    if invalid:
        raise ValueError(
            f"Unsupported VideoToolbox setting(s): {sorted(invalid)}; "
            "use b=<target bits/second>,g=<keyframe interval>, not CQ/CRF/preset"
        )
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"VideoToolbox {key} must be a positive integer")
