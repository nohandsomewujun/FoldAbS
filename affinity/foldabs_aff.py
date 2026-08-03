from typing import Dict

import torch
import torch.nn as nn

try:
    from .foldabs_spec import AffinityHeadConfig, FrozenProtenixAffinityModel
except ModuleNotFoundError as exc:  # compatibility with legacy sys.path setup
    if exc.name != "foldabs_spec":
        raise
    from foldabs_spec import AffinityHeadConfig, FrozenProtenixAffinityModel


class AffinityRegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.mlp(g).reshape(-1, 1)


class FoldAbSAffinityModel(nn.Module):
    """
    Wraps a pretrained FrozenProtenixAffinityModel, freezes it entirely,
    and applies a lightweight regression head on g to predict
    continuous affinity scores.

    Usage:
        base = FrozenProtenixAffinityModel(configs, ckpt_path, ...)
        model = FoldAbSAffinityModel(base, hidden_dim=128)
        optimizer = torch.optim.Adam(model.regression_head.parameters(), lr=1e-3)
    """

    def __init__(
        self,
        base_model: FrozenProtenixAffinityModel,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.freeze_base = freeze_base

        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False

        token_s = base_model.head.to_affinity_pred_score[0].in_features
        self.regression_head = AffinityRegressionHead(
            in_dim=token_s, hidden_dim=hidden_dim, dropout=dropout,
        ).to(base_model.device)

    @property
    def device(self):
        return self.base_model.device

    def forward(
        self, input_feature_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.freeze_base:
            with torch.no_grad():
                base_out = self.base_model(input_feature_dict)
        else:
            base_out = self.base_model(input_feature_dict)

        affinity_score = self.regression_head(base_out["g"])

        return {
            "affinity_score": affinity_score,
            "binding_logits": base_out["logits"],
            "iptm": base_out["iptm"],
        }


__all__ = [
    "AffinityHeadConfig",
    "AffinityRegressionHead",
    "FoldAbSAffinityModel",
    "FrozenProtenixAffinityModel",
]


# Backward-compatible aliases for older internal code paths.
AffinityScoreHead = AffinityRegressionHead
FrozenProtenixAffinityScoreModel = FoldAbSAffinityModel
