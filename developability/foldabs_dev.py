import copy
import functools
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from protenix.model.modules.pairformer import PairformerStack
from protenix.model.protenix import Protenix, update_input_feature_dict


TASK_NAMES = ["aggregation", "expression", "immunogenicity", "thermostability"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_token_pad_mask(input_feature_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "token_pad_mask" in input_feature_dict:
        return input_feature_dict["token_pad_mask"].to(torch.float32)
    n_token = input_feature_dict["token_index"].shape[-1]
    return torch.ones((n_token,), device=input_feature_dict["token_index"].device)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """x: [N, D], mask: [N, 1] -> [D]"""
    masked = x * mask
    denom = mask.sum(dim=0).clamp_min(eps)
    return masked.sum(dim=0) / denom


def _masked_mean_2d(
    z: torch.Tensor, pair_mask: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """z: [N, N, D], pair_mask: [N, N] -> [D]"""
    mask_3d = pair_mask.unsqueeze(-1)
    z_flat = z.reshape(-1, z.shape[-1])
    m_flat = mask_3d.reshape(-1, 1)
    return _masked_mean(z_flat, m_flat, eps)


# ---------------------------------------------------------------------------
# PairwiseConditioning (same as v3/v4)
# ---------------------------------------------------------------------------

class PairwiseConditioning(nn.Module):
    """Fuse pairwise condition features (e.g. distogram) into z_trunk."""

    def __init__(
        self,
        token_z: int,
        dim_cond_feats: int,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(token_z + dim_cond_feats),
            nn.Linear(token_z + dim_cond_feats, token_z, bias=False),
        )
        transitions = []
        hidden_dim = transition_expansion_factor * token_z
        for _ in range(num_transitions):
            transitions.append(
                nn.Sequential(
                    nn.LayerNorm(token_z),
                    nn.Linear(token_z, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, token_z),
                )
            )
        self.transitions = nn.ModuleList(transitions)

    def forward(
        self, z_trunk: torch.Tensor, cond_feats: torch.Tensor
    ) -> torch.Tensor:
        z = torch.cat((z_trunk, cond_feats), dim=-1)
        z = self.proj(z)
        for transition in self.transitions:
            z = transition(z) + z
        return z


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class ESMConfig:
    model_name: str = "esm2_t33_650M_UR50D"
    model_dir: Optional[str] = None
    freeze_esm: bool = True


@dataclass
class DevelopabilityHeadConfig_v5:
    mlp_hidden: int = 512
    mlp_layers: int = 3
    mlp_dropout: float = 0.1
    num_dist_bins: int = 64
    num_transitions: int = 2
    transition_expansion_factor: int = 2


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class DevelopabilityHead_v5(nn.Module):
    """
    Pipeline:
      1. Protenix: distogram -> embed -> PairwiseCond -> PairformerStack -> pool
      2. ESM-2:    sequences -> frozen encode -> mean-pool
      3. Fusion:   concat all pools -> MLP -> scalar
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        esm_dim: int,
        pairformer_cfg: Dict,
        cfg: Optional[DevelopabilityHeadConfig_v5] = None,
        device=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DevelopabilityHeadConfig_v5()

        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        self.distogram_embed = nn.Linear(self.cfg.num_dist_bins, token_z, bias=False)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_cond_feats=token_z,
            num_transitions=self.cfg.num_transitions,
            transition_expansion_factor=self.cfg.transition_expansion_factor,
        )

        self.pairformer_stack = PairformerStack(**pairformer_cfg).to(device)

        # protenix pool: c_s + c_z;  ESM pool: 2 * esm_dim
        mlp_in = token_s + token_z + 2 * esm_dim
        layers: List[nn.Module] = []
        in_dim = mlp_in
        for _ in range(self.cfg.mlp_layers):
            layers.append(nn.Linear(in_dim, self.cfg.mlp_hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.cfg.mlp_dropout))
            in_dim = self.cfg.mlp_hidden
        layers.append(nn.Linear(self.cfg.mlp_hidden, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        distogram_logits: torch.Tensor,
        token_mask: torch.Tensor,
        esm_h_pool: torch.Tensor,
        esm_l_pool: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        s = self.s_norm(torch.clamp(s, min=-512, max=512))

        dist_probs = torch.softmax(distogram_logits.float(), dim=-1)
        dist_embed = self.distogram_embed(dist_probs)

        z = self.z_norm(z)
        z_cond = z + self.pairwise_conditioner(z_trunk=z, cond_feats=dist_embed)

        pair_mask = token_mask[:, None] * token_mask[None, :]
        s, z_cond = self.pairformer_stack(s, z_cond, pair_mask=pair_mask)

        mask_1d = token_mask.unsqueeze(-1)
        s_pool = _masked_mean(s, mask_1d)
        z_pool = _masked_mean_2d(z_cond, pair_mask)

        g = torch.cat([s_pool, z_pool, esm_h_pool, esm_l_pool], dim=-1)
        pred = self.mlp(g).unsqueeze(0)
        return {"pred": pred}


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class FrozenProtenixDevelopabilityModel_v5(nn.Module):
    """
    Frozen Protenix backbone + frozen ESM-2 backbone + trainable fusion head.
    Single-task: one instance per task.
    """

    def __init__(
        self,
        configs,
        checkpoint_path: str,
        esm_cfg: Optional[ESMConfig] = None,
        head_cfg: Optional[DevelopabilityHeadConfig_v5] = None,
        pairformer_overrides: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.configs = configs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_cfg = esm_cfg or ESMConfig()

        # ---- Protenix backbone (frozen) ----
        self.backbone = Protenix(self.configs).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self._freeze_backbone()

        # ---- ESM-2 backbone (frozen) ----
        self._load_esm()

        # ---- Pairformer config ----
        pairformer_cfg = copy.deepcopy(self.configs.model.pairformer)
        pairformer_overrides = pairformer_overrides or {}
        pairformer_cfg.update(pairformer_overrides)

        # ---- Trainable head ----
        self.head = DevelopabilityHead_v5(
            token_s=self.configs.c_s,
            token_z=self.configs.c_z,
            esm_dim=self.esm_dim,
            pairformer_cfg=pairformer_cfg,
            cfg=head_cfg,
            device=self.device,
        ).to(self.device)

    # ---- ESM loading ----

    def _load_esm(self) -> None:
        import esm as _esm

        _orig_load = torch.load
        torch.load = functools.partial(_orig_load, weights_only=False)
        try:
            if self.esm_cfg.model_dir:
                esm_model, alphabet = _esm.pretrained.load_model_and_alphabet_local(
                    self.esm_cfg.model_dir
                )
            else:
                esm_model, alphabet = getattr(_esm.pretrained, self.esm_cfg.model_name)()
        finally:
            torch.load = _orig_load

        self.esm = esm_model.to(self.device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.esm_dim = self.esm.embed_dim

        if self.esm_cfg.freeze_esm:
            for p in self.esm.parameters():
                p.requires_grad = False
            self.esm.eval()

    @torch.no_grad()
    def _encode_esm(self, seq: str) -> torch.Tensor:
        """Encode a single amino acid sequence -> mean-pooled [D_esm]."""
        if not seq:
            return torch.zeros(self.esm_dim, device=self.device)

        data = [("seq", seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        out = self.esm(tokens, repr_layers=[self.esm.num_layers])
        reps = out["representations"][self.esm.num_layers]  # [1, L+2, D]

        padding_idx = self.alphabet.padding_idx
        mask = tokens[:, 1:-1].ne(padding_idx)
        reps = reps[:, 1:-1, :]
        mask_f = mask.unsqueeze(-1).float()
        pooled = (reps * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return pooled.squeeze(0)  # [D_esm]

    # ---- Protenix backbone ----

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"]
        sample_key = next(iter(state_dict))
        if sample_key.startswith("module."):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone.eval()

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _prepare_backbone_inputs(
        self, input_feature_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        input_feature_dict = self.backbone.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)
        return input_feature_dict

    @torch.no_grad()
    def _run_backbone(
        self, input_feature_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_feature_dict = self._prepare_backbone_inputs(input_feature_dict)
        s_inputs, s_trunk, z_trunk = self.backbone.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=self.backbone.N_cycle,
            inplace_safe=True,
            chunk_size=self.configs.infer_setting.chunk_size,
        )
        distogram_logits = self.backbone.distogram_head(z_trunk)
        return s_inputs, s_trunk, z_trunk, distogram_logits

    # ---- Forward ----

    def forward(
        self,
        input_feature_dict: Dict[str, torch.Tensor],
        heavy_seq: str = "",
        light_seq: str = "",
    ) -> Dict[str, torch.Tensor]:
        input_feature_dict = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in input_feature_dict.items()
        }

        _, s_trunk, z_trunk, distogram_logits = self._run_backbone(input_feature_dict)
        token_mask = _build_token_pad_mask(input_feature_dict).to(self.device)

        esm_h_pool = self._encode_esm(heavy_seq)
        esm_l_pool = self._encode_esm(light_seq)

        return self.head(
            s=s_trunk,
            z=z_trunk,
            distogram_logits=distogram_logits,
            token_mask=token_mask,
            esm_h_pool=esm_h_pool,
            esm_l_pool=esm_l_pool,
        )

    def compute_loss(
        self, pred_dict: Dict[str, torch.Tensor], target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        pred = pred_dict["pred"]
        target = target.to(pred.device).view_as(pred)
        mse = nn.functional.mse_loss(pred, target)
        return {"loss": mse, "mse": mse}
