"""
Inference-only dataset for FoldAbS developability prediction.

Mirrors ``spec_inference_dataset.py``'s simplifications (one pass over CSV,
DDP-aware sharding, ``pairedMsaPath`` MSA API, no training-time logic), but
without an antigen chain.

Expected CSV columns:
    sample_id, task, heavy, light,
    heavy_msa_path, light_msa_path
    target  (optional; NaN if running pure inference)

``light`` may be empty / NaN for VHH antibodies; in that case
``light_msa_path`` is not consulted.

Each ``*_msa_path`` entry must point to an existing ``.a3m`` file.
"""

import os
from typing import Dict, Iterator, Optional

import pandas as pd
import torch
from torch.utils.data import IterableDataset

from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.template.template_featurizer import InferenceTemplateFeaturizer
from protenix.data.inference import SampleDictToFeatures
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.torch_utils import dict_to_tensor


TASK_TO_ID = {
    "aggregation": 0,
    "expression": 1,
    "immunogenicity": 2,
    "thermostability": 3,
}
TASK_NAMES = list(TASK_TO_ID.keys())


def _is_missing(val) -> bool:
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(val, str) and not val.strip():
        return True
    return False


def _validate_msa_path(msa_path, field: str) -> str:
    if _is_missing(msa_path):
        raise ValueError(f"{field} is empty; MSA path is required.")
    s = str(msa_path).strip()
    if not s.endswith(".a3m"):
        raise ValueError(f"{field} must be a .a3m file: {s}")
    if not os.path.isfile(s):
        raise FileNotFoundError(f"{field} not found: {s}")
    return s


class DevInferenceDataset(IterableDataset):
    """Inference-only iterable dataset for developability; sharded by DDP rank."""

    REQUIRED_COLUMNS = [
        "sample_id",
        "task",
        "heavy",
        "light",
        "heavy_msa_path",
        "light_msa_path",
    ]

    NAN_TARGET = float("nan")

    def __init__(
        self,
        csv_path: str,
        task: Optional[str] = None,
    ) -> None:
        """
        Args:
            csv_path: Path to the eval CSV.
            task: If given, keep only rows where ``row['task'] == task``.
                  Useful because developability is trained one task at a time.
        """
        super().__init__()
        self.csv_path = csv_path
        self.task_filter = task

        df = pd.read_csv(csv_path)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        if self.task_filter is not None:
            if self.task_filter not in TASK_TO_ID:
                raise ValueError(
                    f"Unknown task '{self.task_filter}'. Choose one of: {TASK_NAMES}"
                )
            df = df[df["task"] == self.task_filter].reset_index(drop=True)

        self.df = df
        self.has_target = "target" in self.df.columns

    def __iter__(self) -> Iterator[Dict]:
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rank = rank * worker_info.num_workers + worker_info.id
            world = world * worker_info.num_workers

        df = self.df
        if world > 1:
            indices = list(range(rank, len(df), world))
            df = df.iloc[indices].reset_index(drop=True)

        for _, row in df.iterrows():
            if self.has_target and not _is_missing(row.get("target")):
                target = float(row["target"])
            else:
                target = self.NAN_TARGET
            sample = self._build_sample(row, target=target)
            sample["sample_id"] = str(row["sample_id"])
            sample["task"] = str(row["task"])
            yield sample

    def _build_sample(self, row: pd.Series, target: float) -> Dict:
        heavy = str(row["heavy"]).upper()
        light = str(row["light"]).upper() if not _is_missing(row.get("light")) else ""
        task_name = str(row["task"])
        if task_name not in TASK_TO_ID:
            raise ValueError(f"Unknown task '{task_name}' in row sample_id={row.get('sample_id')}")
        task_id = TASK_TO_ID[task_name]

        heavy_msa = _validate_msa_path(row.get("heavy_msa_path"), "heavy_msa_path")
        light_msa = _validate_msa_path(row.get("light_msa_path"), "light_msa_path") if light else None

        sequences = [
            {
                "proteinChain": {
                    "sequence": heavy,
                    "count": 1,
                    "unpairedMsaPath": heavy_msa,
                }
            }
        ]
        if light:
            sequences.append(
                {
                    "proteinChain": {
                        "sequence": light,
                        "count": 1,
                        "unpairedMsaPath": light_msa,
                    }
                }
            )

        sample_dict = {
            "name": f"dev_{task_name}_{row.get('sample_id', 'unknown')}",
            "sequences": sequences,
        }

        sample2feat = SampleDictToFeatures(sample_dict)
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()

        msa_features = InferenceMSAFeaturizer.make_msa_feature(
            bioassembly=sample_dict["sequences"],
            atom_array=atom_array,
        )
        if msa_features:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)

        template_features = InferenceTemplateFeaturizer.make_template_feature(
            bioassembly=sequences,
            atom_array=atom_array,
            use_template=False,
            online_template_featurizer=None,
        )

        dummy_feats = []
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            template_features = dict_to_tensor(template_features)
            features_dict.update(template_features)

        if dummy_feats:
            features_dict = make_dummy_feature(
                features_dict=features_dict, dummy_feats=dummy_feats
            )

        feat = data_type_transform(feat_or_label_dict=features_dict)

        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0] if "msa" in feat else 1

        return {
            "input_feature_dict": feat,
            "target": torch.tensor([target], dtype=torch.float32),
            "task_id": torch.tensor([task_id], dtype=torch.long),
            "heavy_seq": heavy,
            "light_seq": light,
            "N_token": torch.tensor([N_token]),
            "N_atom": torch.tensor([N_atom]),
            "N_msa": torch.tensor([N_msa]),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test DevInferenceDataset")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--task", default=None, choices=TASK_NAMES)
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()

    dataset = DevInferenceDataset(csv_path=args.csv_path, task=args.task)
    for i, sample in enumerate(dataset):
        print(
            f"[{i}] sample_id={sample['sample_id']}  task={sample['task']}  "
            f"N_token={int(sample['N_token'])}  "
            f"N_atom={int(sample['N_atom'])}  "
            f"N_msa={int(sample['N_msa'])}  "
            f"target={float(sample['target'])}"
        )
        if i + 1 >= args.n:
            break
