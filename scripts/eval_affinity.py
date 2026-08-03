"""
Multi-GPU via DDP:
    torchrun --nproc_per_node=8 scripts/eval_affinity.py \\
        --config configs/eval_affinity.yml \\
        --foldabs_ckpt /path/to/foldabs_affinity_merged.pt \\
        --csv_path data/aff_eval.csv \\
        --out_csv results/aff_predictions.csv

Single-GPU:
    python scripts/eval_affinity.py \\
        --config configs/eval_affinity.yml \\
        --foldabs_ckpt /path/to/foldabs_affinity_merged.pt \\
        --csv_path data/aff_eval.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROTENIX_DIR = ROOT_DIR / "Protenix"
if PROTENIX_DIR.exists() and str(PROTENIX_DIR) not in sys.path:
    sys.path.insert(0, str(PROTENIX_DIR))

AFFINITY_DIR = ROOT_DIR / "affinity"
if AFFINITY_DIR.exists() and str(AFFINITY_DIR) not in sys.path:
    sys.path.insert(0, str(AFFINITY_DIR))

from affinity.foldabs_aff import (
    AffinityHeadConfig,
    FoldAbSAffinityModel,
    FrozenProtenixAffinityModel,
)
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from datasets.aff_inference_dataloader import get_aff_inference_dataloader
from protenix.config import parse_configs
from protenix.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_configs(model_name: str):
    from ml_collections.config_dict import ConfigDict

    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=f"--model_name {model_name}",
        fill_required_with_null=True,
    )
    model_specific = ConfigDict(model_configs[model_name])
    configs.update(model_specific)

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        cc = major + minor / 10.0
        if 7.0 <= cc < 8.0:
            configs.triangle_multiplicative = "torch"
            configs.triangle_attention = "torch"
            configs.dtype = "fp32"
            print(
                f"[GPU compat] Detected sm_{major}{minor} (cc={cc}). "
                "Forcing torch kernels + fp32."
            )

    return configs


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_config_path(config_path: str, value: str) -> str:
    """Resolve relative checkpoint paths independently of the caller's cwd."""
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return str(path)
    candidates = [Path.cwd() / path, ROOT_DIR / path, Path(config_path).resolve().parent / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((ROOT_DIR / path).resolve())


def _setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if world_size > 1:
        use_gloo = os.environ.get("USE_GLOO_BACKEND", "0") == "1"
        backend = "gloo" if use_gloo else ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30),
        )
        print(f"[Rank {rank}] Process group initialized ({backend})")

    return rank, world_size, device


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


def _extract_prefixed_state(ckpt: Dict, direct_keys: List[str], prefix_keys: List[str]) -> Dict:
    raw_state = ckpt.get("model", ckpt)
    for key in direct_keys:
        if key in ckpt and isinstance(ckpt[key], dict) and ckpt[key]:
            return ckpt[key]
    for prefix in prefix_keys:
        extracted = {
            k[len(prefix):]: v
            for k, v in raw_state.items()
            if k.startswith(prefix)
        }
        if extracted:
            return extracted
    return {}


def _load_foldabs_affinity_checkpoint(
    model: FoldAbSAffinityModel,
    ckpt_path: str,
    device: torch.device,
) -> int:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = int(ckpt.get("step", -1))

    regression_head_state = _extract_prefixed_state(
        ckpt,
        direct_keys=["regression_head"],
        prefix_keys=["regression_head."],
    )
    if not regression_head_state:
        raise ValueError(
            f"{ckpt_path} does not contain a regression head. "
            "Expected a merged FoldAbS affinity checkpoint."
        )
    model.regression_head.load_state_dict(regression_head_state, strict=False)

    affinity_head_state = _extract_prefixed_state(
        ckpt,
        direct_keys=["affinity_head"],
        prefix_keys=["affinity_head."],
    )
    if affinity_head_state:
        model.base_model.head.load_state_dict(affinity_head_state, strict=False)

    return step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for name in ("protenix", "openfold", "unicore"):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="FoldAbS inference: predict continuous affinity score per prot_id."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Inference config (backbone + FoldAbS affinity settings).",
    )
    parser.add_argument(
        "--foldabs_ckpt",
        type=str,
        required=True,
        help="Path to the self-contained FoldAbS affinity checkpoint.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Eval CSV (prot_id, heavy/light/antigen sequences, *_msa_path, [score]).",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--out_csv",
        type=str,
        default=str(ROOT_DIR / "results" / "affinity_predictions.csv"),
        help="Output CSV (prot_id, pred_score[, target]).",
    )
    parser.add_argument("--N_step", type=int, default=None,
                        help="Override N_step_mini_rollout; otherwise use the YAML value.")
    parser.add_argument("--N_sample", type=int, default=None,
                        help="Override N_sample_mini_rollout; otherwise use the YAML value.")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    rank, world_size, device = _setup_distributed()

    seed_everything(seed=args.seed, deterministic=False)
    if _is_main(rank):
        print(f"Seed set to {args.seed}")

    # ---- Load config ----
    cfg = _load_yaml(args.config)
    model_name = cfg.get("model", {}).get("model_name")
    backbone_ckpt = cfg.get("model", {}).get("checkpoint_path")
    if not model_name:
        raise ValueError("model_name not found in config")
    if not backbone_ckpt:
        raise ValueError("checkpoint_path (backbone) not found in config")
    backbone_ckpt = _resolve_config_path(args.config, backbone_ckpt)

    affinity_section = cfg.get("affinity_module", {})
    if args.N_step is not None:
        affinity_section["N_step_mini_rollout"] = args.N_step
    if args.N_sample is not None:
        affinity_section["N_sample_mini_rollout"] = args.N_sample

    head_cfg = AffinityHeadConfig(
        num_dist_bins=int(affinity_section.get("num_dist_bins", 64)),
        max_dist=float(affinity_section.get("max_dist", 22.0)),
        reduce_samples=str(affinity_section.get("reduce_samples", "mean")),
    )

    regression_head_cfg = cfg.get("regression_head", {})
    hidden_dim = int(regression_head_cfg.get("hidden_dim", 256))
    dropout = float(regression_head_cfg.get("dropout", 0.1))
    freeze_base = bool(regression_head_cfg.get("freeze_base", True))

    # ---- Build model ----
    if _is_main(rank):
        print(f"Building model: {model_name}")
        print(f"Backbone checkpoint:  {backbone_ckpt}")
        print(f"FoldAbS checkpoint:   {args.foldabs_ckpt}")
        print(
            f"Diffusion rollout: "
            f"N_sample={affinity_section.get('N_sample_mini_rollout', 1)}, "
            f"N_step={affinity_section.get('N_step_mini_rollout', 20)}"
        )

    configs = _build_configs(model_name)
    base_model = FrozenProtenixAffinityModel(
        configs=configs,
        checkpoint_path=backbone_ckpt,
        head_cfg=head_cfg,
        affinity_cfg=affinity_section,
        device=device,
    )

    model = FoldAbSAffinityModel(
        base_model=base_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_base=freeze_base,
    )

    loaded_step = _load_foldabs_affinity_checkpoint(model, args.foldabs_ckpt, device)
    if _is_main(rank):
        print(f"Loaded FoldAbS affinity checkpoint from step {loaded_step}")

    model.eval()

    # ---- Build dataloader ----
    dataloader = get_aff_inference_dataloader(
        csv_path=args.csv_path,
        batch_size=1,
        num_workers=args.num_workers,
    )
    has_target = dataloader.dataset.has_score

    if _is_main(rank):
        n_total = len(dataloader.dataset.df)
        print(f"Running inference on {n_total} samples with {world_size} GPU(s) "
              f"(targets present: {has_target}) ...")

    # ---- Inference ----
    local_results: List[Dict] = []
    n_done = 0

    with torch.no_grad():
        for batch in dataloader:
            sample = batch if not isinstance(batch, list) else batch[0]
            input_feature_dict = sample["input_feature_dict"]
            prot_id = sample.get("prot_id", "unknown")

            out = model(input_feature_dict)
            pred_score = out["affinity_score"].view(-1)[0].item()

            entry = {"prot_id": prot_id, "pred_score": pred_score}
            if has_target:
                tgt = sample.get("score")
                if tgt is not None:
                    entry["target"] = float(tgt.view(-1)[0].item())
            local_results.append(entry)

            n_done += 1
            if _is_main(rank) and n_done % 10 == 0:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}] rank 0: {n_done} samples done", flush=True)

    if _is_main(rank):
        print(f"Rank 0 finished {n_done} samples")

    # ---- Gather across ranks ----
    if world_size > 1:
        all_results_list = [None] * world_size
        dist.all_gather_object(all_results_list, local_results)
        all_results: List[Dict] = []
        for r in all_results_list:
            all_results.extend(r)
    else:
        all_results = local_results

    if not _is_main(rank):
        _cleanup_distributed()
        return

    # Deduplicate (DDP sharding shouldn't produce dups, but be safe).
    seen = set()
    unique_results = []
    for r in all_results:
        pid = r["prot_id"]
        if pid not in seen:
            seen.add(pid)
            unique_results.append(r)
    all_results = unique_results

    print(f"\nTotal predictions: {len(all_results)}")

    # Preserve input CSV order.
    input_order = pd.read_csv(args.csv_path, usecols=["prot_id"])["prot_id"].astype(str).tolist()
    order_idx = {pid: i for i, pid in enumerate(input_order)}
    all_results.sort(key=lambda r: order_idx.get(str(r["prot_id"]), 10**9))

    columns = ["prot_id", "pred_score"]
    if has_target:
        columns.append("target")
    df_out = pd.DataFrame(all_results, columns=columns)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"Predictions saved to: {args.out_csv}")

    # ---- Optional metrics ----
    if has_target and "target" in df_out.columns:
        mask = df_out["target"].notna() & ~np.isnan(df_out["target"].to_numpy())
        valid = df_out[mask]
        if len(valid) >= 3:
            try:
                from scipy.stats import spearmanr
                preds = valid["pred_score"].to_numpy()
                targets = valid["target"].to_numpy()
                rho = float(spearmanr(preds, targets).correlation)
                mse = float(np.mean((preds - targets) ** 2))
                mae = float(np.mean(np.abs(preds - targets)))
                print(
                    f"Metrics on {len(valid)} samples with targets: "
                    f"spearman={rho:.4f}  mse={mse:.5f}  mae={mae:.5f}"
                )
            except ImportError:
                print("scipy not available; skipping spearman computation.")

    _cleanup_distributed()


if __name__ == "__main__":
    main()
