# RF-DETR MLX inference graph

Adapted from Ak-Gautam/rf-detr-mac, commit
`939362b452d317807c130ae2c82610c564ce2fe2`,
`src/rfdetr/mlx/{config,backbone,layers,ops,model}.py`.
Source: https://github.com/Ak-Gautam/rf-detr-mac
License: Apache-2.0 (see assets/licenses/Apache-2.0.txt).
Original RF-DETR copyright: Roboflow, Inc.

Jasna retains only the inference graph, under its own namespace. The high-level
PIL/NumPy predictor, training imports, checkpoint downloads and generic model
wrappers are excluded. Config is fixed to the actual v6 checkpoint. Attention
uses MLX fused SDPA. The runner strictly loads every checkpoint tensor, maps the
empty keypoint buffer explicitly, and freezes CPU-first reference positional resize
once after loading. No global rfdetr replacement or patch is installed.
