"""Prespecified, single-factor contrasts; E1-E4 form a 2x2 design."""

EXPERIMENTS = {
    "E0": dict(use_gatv2=False, semantic_mode="global", fusion_mode="none", fusion_alpha=0.6),
    "E1": dict(use_gatv2=True, semantic_mode="global", fusion_mode="none", fusion_alpha=0.6),
    "E2": dict(use_gatv2=True, semantic_mode="node", fusion_mode="none", fusion_alpha=0.6),
    "E3": dict(use_gatv2=True, semantic_mode="global", fusion_mode="cross", fusion_alpha=0.6),
    "E4": dict(use_gatv2=True, semantic_mode="node", fusion_mode="cross", fusion_alpha=0.6),
    "E5": dict(use_gatv2=True, semantic_mode="node", fusion_mode="cross", fusion_alpha=1.0),
    "E6": dict(use_gatv2=True, semantic_mode="node", fusion_mode="pointwise", fusion_alpha=0.6),
    "E7": dict(use_gatv2=False, semantic_mode="node", fusion_mode="cross", fusion_alpha=0.6),
}

CONTRASTS = {
    "gatv2_without_fusion": ("E1", "E0"),
    "node_weights_only": ("E2", "E1"),
    "node_weights_after_mixing": ("E4", "E3"),
    "cross_view_with_global_weights": ("E3", "E1"),
    "cross_view_with_node_weights": ("E4", "E2"),
    "retain_encoded_representation": ("E4", "E5"),
    "cross_view_vs_equal_parameter_mlp": ("E4", "E6"),
    "gatv2_after_fusion": ("E4", "E7"),
    "fusion_on_original_gat": ("E7", "E0"),
}

METAPATHS = ["gg", "ga-ag", "gb-bg", "gc-cg", "gd-dg", "ge-eg", "gf-fg", "gh-hg", "gi-ig"]
