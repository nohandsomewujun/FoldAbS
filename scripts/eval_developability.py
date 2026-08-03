"""
Multi-GPU via DDP:
    torchrun --nproc_per_node=8 scripts/eval_developability.py \\
        --config configs/eval_developability.yml \\
        --head_ckpt checkpoints/aggre.pt \\
        --csv_path data/developability/test.csv \\
        --task aggregation \\
        --out_csv results/dev_aggregation.csv

Single-GPU:
    python scripts/eval_developability.py \\
        --config configs/eval_developability.yml \\
        --head_ckpt checkpoints/aggre.pt \\
        --csv_path data/developability/test.csv \\
        --task aggregation
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

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

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from datasets.dev_inference_dataloader import get_dev_inference_dataloader
from datasets.dev_inference_dataset import TASK_NAMES
from developability.foldabs_dev import (
    DevelopabilityHeadConfig_v5,
    ESMConfig,
    FrozenProtenixDevelopabilityModel_v5,
)
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


def _zscore_to_raw(z: float, stats: Dict) -> float:
    raw = z * float(stats["zscore_std"]) + float(stats["zscore_mean"])
    if stats.get("log_transform", False):
        raw = float(np.exp(raw))
    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for name in ("protenix", "openfold", "unicore"):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="FoldAbS inference: predict developability per sample_id (one task)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training config used for this task "
             "(provides model_name, backbone ckpt, ESM cfg, head cfg, pairformer cfg).",
    )
    parser.add_argument(
        "--head_ckpt",
        type=str,
        required=True,
        help="Path to a released developability head checkpoint (key ``head``).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Eval CSV (sample_id, task, heavy, light, *_msa_path, [target]).",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=TASK_NAMES,
        help="Which task to evaluate.",
    )
    parser.add_argument(
        "--norm_stats",
        type=str,
        default=None,
        help="Optional path to norm_stats.json. If given, raw-scale predictions "
             "are added via inverse z-score (+ optional log_transform).",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to results/dev_{task}.csv.",
    )
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    rank, world_size, device = _setup_distributed()

    seed_everything(seed=args.seed, deterministic=False)
    if _is_main(rank):
        print(f"Seed set to {args.seed}")

    # ---- Load config ----
    cfg = _load_yaml(args.config)
    model_name = cfg["model"]["model_name"]
    backbone_ckpt = cfg["model"]["checkpoint_path"]
    if not model_name:
        raise ValueError("model_name not found in config")
    if not backbone_ckpt:
        raise ValueError("checkpoint_path (backbone) not found in config")
    backbone_ckpt = _resolve_config_path(args.config, backbone_ckpt)

    # ---- Resolve task ----
    head_ckpt = torch.load(args.head_ckpt, map_location=device, weights_only=False)
    task_name: str = args.task
    if task_name not in TASK_NAMES:
        raise ValueError(f"Unknown task '{task_name}'. Choose from: {TASK_NAMES}")

    # ---- Head / ESM / Pairformer configs ----
    mlp_cfg = cfg.get("developability_module", {}).get("mlp", {})
    cond_cfg = cfg.get("developability_module", {}).get("pairwise_conditioning", {})
    pf_cfg = cfg.get("developability_module", {}).get("pairformer", {})
    esm_module_cfg = cfg.get("esm_module", {})
    if esm_module_cfg.get("model_dir"):
        esm_module_cfg["model_dir"] = _resolve_config_path(
            args.config, esm_module_cfg["model_dir"]
        )

    head_cfg_obj = DevelopabilityHeadConfig_v5(
        mlp_hidden=mlp_cfg.get("hidden_dim", 512),
        mlp_layers=mlp_cfg.get("num_layers", 3),
        mlp_dropout=mlp_cfg.get("dropout", 0.1),
        num_dist_bins=cond_cfg.get("num_dist_bins", 64),
        num_transitions=cond_cfg.get("num_transitions", 2),
        transition_expansion_factor=cond_cfg.get("transition_expansion_factor", 2),
    )
    esm_cfg = ESMConfig(
        model_name=esm_module_cfg.get("model_name", "esm2_t33_650M_UR50D"),
        model_dir=esm_module_cfg.get("model_dir"),
        freeze_esm=esm_module_cfg.get("freeze_esm", True),
    )
    pairformer_overrides = {
        "n_blocks": pf_cfg.get("n_blocks", 1),
        "dropout": pf_cfg.get("dropout", 0.1),
    }
    if pf_cfg.get("blocks_per_ckpt") is not None:
        pairformer_overrides["blocks_per_ckpt"] = pf_cfg["blocks_per_ckpt"]

    # ---- Build model ----
    if _is_main(rank):
        print(f"Task               : {task_name}")
        print(f"Model              : {model_name}")
        print(f"Backbone checkpoint: {backbone_ckpt}")
        print(f"Head checkpoint    : {args.head_ckpt}")
        print(f"ESM                : {esm_cfg.model_name} (frozen={esm_cfg.freeze_esm})")
        print(f"Norm stats         : {args.norm_stats or '(none, z-score only)'}")

    configs = _build_configs(model_name)
    model = FrozenProtenixDevelopabilityModel_v5(
        configs=configs,
        checkpoint_path=backbone_ckpt,
        esm_cfg=esm_cfg,
        head_cfg=head_cfg_obj,
        pairformer_overrides=pairformer_overrides,
        device=device,
    )

    if "head" not in head_ckpt:
        raise ValueError(
            f"--head_ckpt {args.head_ckpt} does not contain a 'head' key. "
            "Expected a checkpoint saved by train_developability_v5.py."
        )
    model.head.load_state_dict(head_ckpt["head"])
    if _is_main(rank):
        step = head_ckpt.get("step", "?")
        print(f"Loaded head from step {step}")

    model.eval()

    # ---- Norm stats ----
    norm_stats = None
    task_stats = None
    if args.norm_stats:
        with open(args.norm_stats, "r", encoding="utf-8") as f:
            norm_stats = json.load(f)
        if task_name not in norm_stats:
            raise ValueError(f"task '{task_name}' missing in norm_stats: {list(norm_stats)}")
        task_stats = norm_stats[task_name]

    # ---- Build dataloader ----
    dataloader = get_dev_inference_dataloader(
        csv_path=args.csv_path,
        task=task_name,
        batch_size=1,
        num_workers=args.num_workers,
    )
    has_target = dataloader.dataset.has_target

    if _is_main(rank):
        n_total = len(dataloader.dataset.df)
        print(f"Running inference on {n_total} samples (task={task_name}) "
              f"with {world_size} GPU(s) (targets present: {has_target}) ...")

    # ---- Inference ----
    local_results: List[Dict] = []
    n_done = 0

    with torch.no_grad():
        for batch in dataloader:
            sample = batch if not isinstance(batch, list) else batch[0]
            input_feat = sample["input_feature_dict"]
            heavy_seq = sample.get("heavy_seq", "")
            light_seq = sample.get("light_seq", "")
            sample_id = sample.get("sample_id", "unknown")

            pred_dict = model(input_feat, heavy_seq=heavy_seq, light_seq=light_seq)
            z_pred = pred_dict["pred"].view(-1)[0].item()

            entry = {"sample_id": sample_id, "task": task_name, "pred": z_pred}
            if task_stats is not None:
                entry["pred_raw"] = _zscore_to_raw(z_pred, task_stats)
            if has_target:
                tgt = sample.get("target")
                if tgt is not None:
                    z_tgt = float(tgt.view(-1)[0].item())
                    entry["target"] = z_tgt
                    if task_stats is not None:
                        entry["target_raw"] = _zscore_to_raw(z_tgt, task_stats)
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

    # Deduplicate
    seen = set()
    unique_results = []
    for r in all_results:
        sid = r["sample_id"]
        if sid not in seen:
            seen.add(sid)
            unique_results.append(r)
    all_results = unique_results

    print(f"\nTotal predictions: {len(all_results)}")

    # Preserve input CSV order (within the selected task).
    df_in = pd.read_csv(args.csv_path, usecols=["sample_id", "task"])
    df_in = df_in[df_in["task"] == task_name].reset_index(drop=True)
    input_order = df_in["sample_id"].astype(str).tolist()
    order_idx = {sid: i for i, sid in enumerate(input_order)}
    all_results.sort(key=lambda r: order_idx.get(str(r["sample_id"]), 10**9))

    columns = ["sample_id", "task", "pred"]
    if task_stats is not None:
        columns.append("pred_raw")
    if has_target:
        columns.append("target")
        if task_stats is not None:
            columns.append("target_raw")
    df_out = pd.DataFrame(all_results, columns=columns)

    out_csv = args.out_csv or str(ROOT_DIR / "results" / f"dev_{task_name}.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Predictions saved to: {out_csv}")

    # ---- Optional metrics ----
    if has_target and "target" in df_out.columns:
        valid = df_out.dropna(subset=["target"])
        if len(valid) >= 3:
            try:
                from scipy.stats import spearmanr
                z_preds = valid["pred"].to_numpy()
                z_targets = valid["target"].to_numpy()
                rho_z = float(spearmanr(z_preds, z_targets).correlation)
                msg = (f"Metrics on {len(valid)} samples (task={task_name}): "
                       f"spearman_z={rho_z:.4f}")
                if task_stats is not None and "target_raw" in df_out.columns:
                    raw_preds = valid["pred_raw"].to_numpy()
                    raw_targets = valid["target_raw"].to_numpy()
                    rho_raw = float(spearmanr(raw_preds, raw_targets).correlation)
                    msg += f"  spearman_raw={rho_raw:.4f}"
                print(msg)
            except ImportError:
                print("scipy not available; skipping spearman computation.")

    _cleanup_distributed()


if __name__ == "__main__":
    main()
