"""
Supports multi-GPU via DDP:
    torchrun --nproc_per_node=8 scripts/eval_spec.py \
        --config xxx.yml \
        --spec_ckpt xxx.pt \
        --csv_path xxx.csv \
        --out_csv xxx.csv

Single-GPU:
    python scripts/eval_spec.py \
        --config xxx.yml \
        --spec_ckpt xxx.pt \
        --csv_path xxx.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

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

from affinity.foldabs_spec import AffinityHeadConfig, FrozenProtenixAffinityModel
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from datasets.spec_inference_dataloader import get_spec_inference_dataloader
from protenix.config import parse_configs
from protenix.utils.seed import seed_everything


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for name in ("protenix", "openfold", "unicore"):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="FoldAbS inference: predict binding probability per prot_id."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Inference YAML config (model and affinity_module settings).",
    )
    parser.add_argument(
        "--spec_ckpt", "--affinity_ckpt",
        dest="spec_ckpt",
        type=str,
        required=True,
        help="Path to the specificity head checkpoint (legacy alias: --affinity_ckpt).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Eval CSV (with prot_id, heavy/light/antigen sequences and *_msa_path).",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--out_csv",
        type=str,
        default=str(ROOT_DIR / "results" / "predictions.csv"),
        help="Output CSV (two columns: prot_id, prob).",
    )
    parser.add_argument(
        "--N_step",
        type=int,
        default=None,
        help="Override N_step_mini_rollout; otherwise use the YAML value.",
    )
    parser.add_argument(
        "--N_sample",
        type=int,
        default=None,
        help="Override N_sample_mini_rollout; otherwise use the YAML value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Random seed (default: 101, matching official Protenix).",
    )
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

    # ---- Build model ----
    if _is_main(rank):
        print(f"Building model: {model_name}")
        print(f"Backbone checkpoint: {backbone_ckpt}")
        print(f"Specificity head checkpoint: {args.spec_ckpt}")
        print(
            f"Diffusion rollout: "
            f"N_sample={affinity_section.get('N_sample_mini_rollout', 1)}, "
            f"N_step={affinity_section.get('N_step_mini_rollout', 20)}"
        )

    configs = _build_configs(model_name)
    model = FrozenProtenixAffinityModel(
        configs=configs,
        checkpoint_path=backbone_ckpt,
        head_cfg=head_cfg,
        affinity_cfg=affinity_section,
        device=device,
    )

    ckpt = torch.load(args.spec_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    if _is_main(rank):
        step = ckpt.get("step", "?")
        print(f"Loaded specificity head from step {step}")

    model.eval()

    # ---- Build dataloader ----
    dataloader = get_spec_inference_dataloader(
        csv_path=args.csv_path,
        batch_size=1,
        num_workers=args.num_workers,
    )

    if _is_main(rank):
        n_total = len(dataloader.dataset.df)
        print(f"Running inference on {n_total} samples with {world_size} GPU(s) ...")

    # ---- Inference ----
    local_results: List[Dict] = []
    n_done = 0

    with torch.no_grad():
        for batch in dataloader:
            sample = batch if not isinstance(batch, list) else batch[0]
            input_feature_dict = sample["input_feature_dict"]
            prot_id = sample.get("prot_id", "unknown")

            out = model(input_feature_dict)
            logits = out["logits"]  # [1, 1]
            prob = torch.sigmoid(logits).item()

            local_results.append({"prot_id": prot_id, "prob": prob})

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

    # Preserve input CSV order for readability.
    input_order = pd.read_csv(args.csv_path, usecols=["prot_id"])["prot_id"].astype(str).tolist()
    order_idx = {pid: i for i, pid in enumerate(input_order)}
    all_results.sort(key=lambda r: order_idx.get(str(r["prot_id"]), 10**9))

    df = pd.DataFrame(all_results, columns=["prot_id", "prob"])
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Predictions saved to: {args.out_csv}")

    _cleanup_distributed()


if __name__ == "__main__":
    main()
