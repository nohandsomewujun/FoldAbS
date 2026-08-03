import copy
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from protenix.model.modules.pairformer import PairformerStack
from protenix.model.protenix import Protenix, update_input_feature_dict
from protenix.model.sample_confidence import calculate_iptm, calculate_ptm
from protenix.model.triangular.layers import LayerNorm

"""
def cuda_mem(tag=""):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    free, total = torch.cuda.mem_get_info()
    free_mb = free / 1024**2
    total_mb = total / 1024**2
    print(
        f"[{tag}] alloc={alloc:.1f}MB peak={peak:.1f}MB "
        f"free={free_mb:.1f}MB total={total_mb:.1f}MB"
    )
"""

class PairwiseConditioning(nn.Module):
    def __init__(
        self,
        token_z: int,
        dim_token_rel_pos_feats: int,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
    ) -> None:
        super().__init__()
        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(token_z + dim_token_rel_pos_feats),
            nn.Linear(token_z + dim_token_rel_pos_feats, token_z, bias=False),
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
        self, z_trunk: torch.Tensor, token_rel_pos_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        if z_trunk.dim() == 3:
            z_trunk = z_trunk.unsqueeze(0)
        if token_rel_pos_feats.dim() == 3:
            token_rel_pos_feats = token_rel_pos_feats.unsqueeze(0)
        """
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)
        for transition in self.transitions:
            z = transition(z) + z
        return z


@dataclass
class AffinityHeadConfig:
    num_dist_bins: int = 64
    max_dist: float = 22.0
    reduce_samples: str = "mean"  # "mean" or "first"


def _build_token_pad_mask(input_feature_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "token_pad_mask" in input_feature_dict:
        return input_feature_dict["token_pad_mask"].to(torch.float32)
    n_token = input_feature_dict["token_index"].shape[-1]
    return torch.ones((n_token,), device=input_feature_dict["token_index"].device)


def _build_token_to_rep_atom(
    input_feature_dict: Dict[str, torch.Tensor]
) -> torch.Tensor:
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"].long()
    rep_atom_mask = input_feature_dict["distogram_rep_atom_mask"].bool()
    if atom_to_token_idx.dim() == 2 and atom_to_token_idx.shape[0] == 1:
        atom_to_token_idx = atom_to_token_idx.squeeze(0)
    if rep_atom_mask.dim() == 2 and rep_atom_mask.shape[0] == 1:
        rep_atom_mask = rep_atom_mask.squeeze(0)

    n_token = int(input_feature_dict["token_index"].shape[-1])
    n_atom = int(atom_to_token_idx.shape[-1])

    rep_atom_idx = torch.full((n_token,), -1, dtype=torch.long)
    rep_atom_positions = torch.nonzero(rep_atom_mask, as_tuple=False).view(-1)
    rep_token_idx = atom_to_token_idx[rep_atom_mask]

    # Fill one rep atom per token (first occurrence)
    rep_atom_positions_cpu = rep_atom_positions.detach().cpu().tolist()
    rep_token_idx_cpu = rep_token_idx.detach().cpu().tolist()
    for atom_idx, token_idx in zip(rep_atom_positions_cpu, rep_token_idx_cpu):
        if rep_atom_idx[token_idx] == -1:
            rep_atom_idx[token_idx] = atom_idx

    rep_atom_idx = rep_atom_idx.to(atom_to_token_idx.device)
    token_to_rep_atom = torch.zeros(
        (n_token, n_atom), device=atom_to_token_idx.device, dtype=torch.float32
    )
    valid = rep_atom_idx >= 0
    token_to_rep_atom[valid, rep_atom_idx[valid]] = 1.0
    return token_to_rep_atom


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # x: [N, D], mask: [N, 1]
    masked = x * mask
    denom = mask.sum(dim=0).clamp_min(eps)
    return masked.sum(dim=0) / denom


def _build_cross_pair_mask(
    input_feature_dict: Dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor:
    pad_token_mask = _build_token_pad_mask(input_feature_dict).to(device)
    lig_mask = input_feature_dict["affinity_token_mask"].to(torch.bool).to(device)
    lig_mask = lig_mask * pad_token_mask
    rec_mask = (1.0 - lig_mask) * pad_token_mask

    cross_pair_mask = (
        lig_mask[:, None] * rec_mask[None, :]
        + rec_mask[:, None] * lig_mask[None, :]
        + lig_mask[:, None] * lig_mask[None, :]
    )
    return cross_pair_mask


class BoltzStyleAffinityHead(nn.Module):
    def __init__(
        self,
        token_z: int,
        token_s: int,
        cfg: Optional[AffinityHeadConfig] = None,
        pairformer_cfg: Optional[Dict] = None,
        b_pae: int = 64,
        device=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or AffinityHeadConfig()
        boundaries = torch.linspace(2.0, self.cfg.max_dist, self.cfg.num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(self.cfg.num_dist_bins, token_z)

        # PAE prob -> pairwise embedding (frozen PAE logits from confidence head)
        self.pae_pairwise_embed = nn.Linear(b_pae, token_z, bias=False)

        self.z_norm = nn.LayerNorm(token_z)
        self.z_linear = nn.Linear(token_z, token_z, bias=False)
        # dim_token_rel_pos_feats = token_z (distogram) + token_z (pae)
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z * 2,
            num_transitions=2,
        )

        # iPTM feature embedding: scalar -> small vector
        iptm_embed_dim = 32
        self.iptm_embed = nn.Sequential(
            nn.Linear(1, iptm_embed_dim),
            nn.ReLU(),
            nn.Linear(iptm_embed_dim, iptm_embed_dim),
        )

        mlp_in = token_z + 2 * token_s + iptm_embed_dim
        self.affinity_out_mlp = nn.Sequential(
            nn.Linear(mlp_in, token_s),
            nn.ReLU(),
            nn.Linear(token_s, token_s),
            nn.ReLU(),
        )
        self.to_affinity_pred_score = nn.Sequential(
            nn.Linear(token_s, token_s),
            nn.ReLU(),
            nn.Linear(token_s, token_s),
            nn.ReLU(),
            nn.Linear(token_s, 1),
        )
        self.to_affinity_logits_binary = nn.Linear(1, 1)

        # Learnable residual bypass: iPTM -> logit shortcut
        self.iptm_bypass = nn.Linear(1, 1, bias=True)
        # Sample-dependent gate: sees pure structural features + iPTM
        # to decide per-sample whether to trust head or iPTM.
        # Input: token_z + 2*token_s (g_struct, no iPTM contamination) + 1 (iPTM)
        gate_in_dim = token_z + 2 * token_s + 1
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.input_strunk_ln = LayerNorm(token_s)

        self.pairformer_stacks = PairformerStack(**pairformer_cfg).to(device)

    def forward(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        x_pred: torch.Tensor,
        feats: Dict[str, torch.Tensor],
        pae_logits: torch.Tensor,
        iptm_score: float = 0.0,
    ) -> torch.Tensor:
        # z: [N, N, Cz], s: [N, Cs], x_pred: [1, N_atom, 3]
        # pae_logits: [N_token, N_token, b_pae]  (from best sample's confidence head)
        # iptm_score: scalar iPTM value for this sample
        token_to_rep_atom = _build_token_to_rep_atom(feats)
        token_to_rep_atom = token_to_rep_atom.to(z.device)

        x_pred_repr = torch.matmul(token_to_rep_atom.float(), x_pred.squeeze(0)) # [N_tokens, 3]
        d = torch.cdist(x_pred_repr, x_pred_repr) # [N_tokens, N_tokens]
        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long() # [N_tokens, N_tokens, N_bins]

        distogram = self.dist_bin_pairwise_embed(distogram)  # [N_tokens, N_tokens, token_z]

        # PAE prob → pairwise embedding
        pae_prob = torch.softmax(pae_logits.detach().float(), dim=-1)  # [N_token, N_token, b_pae]
        pae_embed = self.pae_pairwise_embed(pae_prob)  # [N_token, N_token, token_z]

        # Concatenate distogram + pae as pairwise condition features
        pairwise_feats = torch.cat([distogram, pae_embed], dim=-1)  # [N_token, N_token, token_z*2]

        z = self.z_linear(self.z_norm(z))
        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=pairwise_feats)

        pad_token_mask = _build_token_pad_mask(feats).to(z.device).unsqueeze(-1)
        lig_mask = feats["affinity_token_mask"].to(torch.bool).to(z.device).unsqueeze(-1)
        lig_mask = lig_mask * pad_token_mask
        rec_mask = (1.0 - lig_mask) * pad_token_mask

        s = self.input_strunk_ln(torch.clamp(s, min=-512, max=512))

        cross_pair_mask = _build_cross_pair_mask(feats, z.device) # [N_tokens, N_tokens]
        s, z = self.pairformer_stacks(s, z, pair_mask=cross_pair_mask)

        # Remove self-interactions for pooling (out-of-place to avoid autograd inplace error)
        pool_mask = cross_pair_mask * (1 - torch.eye(lig_mask.shape[0], device=lig_mask.device))

        g_z = (z * pool_mask[..., None]).sum((0, 1)) / (pool_mask.sum((0, 1)) + 1e-7)

        s_lig = _masked_mean(s, lig_mask)
        s_rec = _masked_mean(s, rec_mask)

        # Pure structural features (before iPTM injection) for gate
        g_struct = torch.cat([g_z, s_lig, s_rec], dim=-1)  # [Cz + 2*Cs]

        # iPTM feature injection: embed scalar iPTM into a learned vector
        iptm_tensor = torch.tensor([[iptm_score]], device=z.device, dtype=z.dtype)
        iptm_feat = self.iptm_embed(iptm_tensor).squeeze(0)  # [iptm_embed_dim]

        g = torch.cat([g_struct, iptm_feat], dim=-1)  # [Cz + 2*Cs + iptm_embed_dim]

        g = self.affinity_out_mlp(g)
        affinity_pred_score = self.to_affinity_pred_score(g).reshape(-1, 1)
        head_logits = self.to_affinity_logits_binary(affinity_pred_score).reshape(-1, 1)

        # Learnable residual bypass: blend head output with iPTM shortcut
        iptm_logit = self.iptm_bypass(iptm_tensor)  # [1, 1]
        # Sample-dependent gate: pure structural features (no iPTM) + iPTM
        # gate sees "what structure says" vs "what iPTM says" independently
        gate_input = torch.cat([g_struct.unsqueeze(0), iptm_tensor], dim=-1)  # [1, Cz+2Cs+1]
        gate = torch.sigmoid(self.gate_net(gate_input))  # [1, 1], per-sample
        affinity_logits_binary = gate * head_logits + (1.0 - gate) * iptm_logit

        return affinity_logits_binary, g


class FrozenProtenixAffinityModel(nn.Module):
    def __init__(
        self,
        configs,
        checkpoint_path: str,
        head_cfg: Optional[AffinityHeadConfig] = None,
        affinity_cfg: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.configs = configs
        self.affinity_cfg = affinity_cfg or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = Protenix(self.configs).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self._freeze_backbone()

        pairformer_cfg = copy.deepcopy(self.configs.model.pairformer)
        if affinity_cfg and isinstance(affinity_cfg.get("pairformer"), dict):
            pairformer_cfg.update(affinity_cfg["pairformer"])

        b_pae = affinity_cfg.get("b_pae", 64) if affinity_cfg else 64
        self.head = BoltzStyleAffinityHead(
            token_z=self.configs.c_z,
            token_s=self.configs.c_s,
            cfg=head_cfg,
            pairformer_cfg=pairformer_cfg,
            b_pae=b_pae,
            device=self.device,
        ).to(self.device)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        sample_key = [k for k in checkpoint["model"].keys()][0]
        if sample_key.startswith("module."):
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.backbone.load_state_dict(checkpoint["model"], strict=True)
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

    def _run_backbone(
        self, input_feature_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_feature_dict = self._prepare_backbone_inputs(input_feature_dict)
        with torch.no_grad():
            s_inputs, s_trunk, z_trunk = self.backbone.get_pairformer_output(
                input_feature_dict=input_feature_dict,
                N_cycle=self.backbone.N_cycle,
                inplace_safe=True,
                chunk_size=self.configs.infer_setting.chunk_size,
            )

            # mini rollout: generate N diffusion samples
            N_sample_mini_rollout = self.affinity_cfg.get("N_sample_mini_rollout", 1)
            N_step_mini_rollout = self.affinity_cfg.get("N_step_mini_rollout", 20)

            noise_schedule = self.backbone.inference_noise_scheduler(
                N_step=N_step_mini_rollout,
                device=s_inputs.device,
                dtype=s_inputs.dtype,
            )

            x_pred = self.backbone.sample_diffusion(
                denoise_net=self.backbone.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs.detach(),
                s_trunk=s_trunk.detach(),
                z_trunk=z_trunk.detach(),
                pair_z=None,
                p_lm=None,
                c_l=None,
                N_sample=N_sample_mini_rollout,
                noise_schedule=noise_schedule,
                enable_efficient_fusion=self.backbone.enable_efficient_fusion,
            )  # [N_sample, N_atom, 3]

            # --- Run frozen confidence head to get PAE logits ---
            _, pae_preds, _, _ = self.backbone.confidence_head(
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_mask=None,
                x_pred_coords=x_pred,
            )  # pae_preds: [N_sample, N_token, N_token, b_pae]

            # --- Select best sample by ranking_score (matching Protenix official) ---
            # ranking_score = 0.8 * iPTM + 0.2 * pTM  (clash penalty skipped)
            pae_prob = torch.softmax(pae_preds.float(), dim=-1)
            bin_params = dict(min_bin=0, max_bin=32, no_bins=64)
            iptm_scores = calculate_iptm(
                pae_prob,
                has_frame=input_feature_dict["has_frame"],
                asym_id=input_feature_dict["asym_id"],
                **bin_params,
            )  # [N_sample]
            ptm_scores = calculate_ptm(
                pae_prob,
                has_frame=input_feature_dict["has_frame"],
                **bin_params,
            )  # [N_sample]
            ranking_scores = 0.8 * iptm_scores + 0.2 * ptm_scores  # [N_sample]

            best_idx = ranking_scores.argmax()
            best_iptm = iptm_scores[best_idx].item()
            x_pred_best = x_pred[best_idx : best_idx + 1]      # [1, N_atom, 3]
            pae_logits_best = pae_preds[best_idx]               # [N_token, N_token, b_pae]
            del pae_prob, pae_preds

        return s_trunk, s_inputs, z_trunk, x_pred_best.detach(), pae_logits_best.detach(), best_iptm

    def _run_pairformer_cross(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        input_feature_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cross_pair_mask = _build_cross_pair_mask(input_feature_dict, z_trunk.device)
        s_refined, z_refined = self.affinity_pairformer(
            s_trunk,
            z_trunk,
            pair_mask=cross_pair_mask,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=True,
            chunk_size=self.configs.infer_setting.chunk_size,
        )
        return s_refined, z_refined

    def forward(self, input_feature_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        input_feature_dict = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in input_feature_dict.items()
        }
        if "affinity_token_mask" not in input_feature_dict:
            raise KeyError("Missing affinity_token_mask in input_feature_dict")

        s_trunk, s_inputs, z_trunk, x_pred, pae_logits, iptm = self._run_backbone(input_feature_dict)

        logits, g = self.head(
            z=z_trunk,
            s=s_trunk,
            x_pred=x_pred,
            feats=input_feature_dict,
            pae_logits=pae_logits,
            iptm_score=iptm,
        )

        return {"logits": logits, "g": g, "iptm": iptm}
