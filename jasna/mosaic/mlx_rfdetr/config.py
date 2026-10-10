# Adapted from Ak-Gautam/rf-detr-mac (Apache-2.0), commit 939362b.
# See NOTICE.md for provenance and Jasna-specific changes.
"""Inference-only MLX model configs for the Mac-only RF-DETR fork."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MLXModelConfig:
    """Static configuration required to run an MLX inference graph."""

    name: str
    num_classes: int
    resolution: int
    patch_size: int
    hidden_dim: int
    backbone_dim: int
    backbone_heads: int
    backbone_layers: int
    backbone_mlp_ratio: int
    dec_layers: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    num_windows: int
    num_queries: int
    num_query_embeddings: int
    num_select: int
    positional_encoding_size: int
    projector_scales: tuple[str, ...]
    out_feature_indexes: tuple[int, ...]
    group_detr: int = 13
    num_register_tokens: int = 0
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    bbox_reparam: bool = True
    two_stage: bool = True
    lite_refpoint_refine: bool = True
    license: str = "Apache-2.0"
    checkpoint_name: str = ""


JasnaV6MLXConfig = MLXModelConfig(
    name="jasna-rfdetr-v6",
    num_classes=2,
    resolution=576,
    patch_size=12,
    hidden_dim=256,
    backbone_dim=384,
    backbone_heads=6,
    backbone_layers=12,
    backbone_mlp_ratio=4,
    dec_layers=5,
    sa_nheads=8,
    ca_nheads=16,
    dec_n_points=2,
    num_windows=2,
    num_queries=200,
    num_query_embeddings=200,
    num_select=16,
    positional_encoding_size=36,
    projector_scales=("P4",),
    out_feature_indexes=(3, 6, 9, 12),
    segmentation_head=True,
)
