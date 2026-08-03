"""
Inference-only dataset for FoldAbS affinity score regression.

Mirrors ``spec_inference_dataset.py`` (same antibody-antigen input pipeline +
``pairedMsaPath`` MSA API) but expects a continuous ``score`` column instead
of a binary ``label``.

Expected CSV columns:
    prot_id, heavy_sequence, light_sequence, antigen_sequence,
    heavy_msa_path, light_msa_path, antigen_msa_path
    score   (optional; NaN-tolerant; absent means inference-only)

Each ``*_msa_path`` entry must point to an existing ``.a3m`` file.
"""

import os
from typing import Dict, Iterator

import pandas as pd
import torch
from torch.utils.data import IterableDataset

from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.template.template_featurizer import InferenceTemplateFeaturizer
from protenix.data.inference import SampleDictToFeatures
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.torch_utils import dict_to_tensor


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


class AffInferenceDataset(IterableDataset):
    """Inference-only iterable dataset for affinity regression; sharded by DDP."""

    REQUIRED_COLUMNS = [
        "prot_id",
        "heavy_sequence",
        "light_sequence",
        "antigen_sequence",
        "heavy_msa_path",
        "light_msa_path",
        "antigen_msa_path",
    ]

    NAN_SCORE = float("nan")

    def __init__(self, csv_path: str) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        self.has_score = "score" in self.df.columns

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
            if self.has_score and not _is_missing(row.get("score")):
                score = float(row["score"])
            else:
                score = self.NAN_SCORE
            sample = self._build_sample(row, score=score)
            sample["prot_id"] = str(row["prot_id"])
            yield sample

    def _build_sample(self, row: pd.Series, score: float) -> Dict:
        heavy = str(row["heavy_sequence"])
        light = str(row["light_sequence"]) if pd.notna(row["light_sequence"]) else ""
        antigen = str(row["antigen_sequence"])

        heavy_msa = _validate_msa_path(row.get("heavy_msa_path"), "heavy_msa_path")
        light_msa = _validate_msa_path(row.get("light_msa_path"), "light_msa_path") if light else None
        antigen_msa = _validate_msa_path(row.get("antigen_msa_path"), "antigen_msa_path")

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
        sequences.append(
            {
                "proteinChain": {
                    "sequence": antigen,
                    "count": 1,
                    "unpairedMsaPath": antigen_msa,
                }
            }
        )

        sample_dict = {"name": str(row["prot_id"]), "sequences": sequences}

        sample2feat = SampleDictToFeatures(sample_dict)
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()

        entity_id = features_dict.get("entity_id")
        if entity_id is not None:
            ligand_entity_ids = [0]
            if light:
                ligand_entity_ids.append(1)
            affinity_mask = torch.zeros_like(entity_id, dtype=torch.float32)
            for ent_id in ligand_entity_ids:
                affinity_mask = affinity_mask + (entity_id == ent_id).float()
            features_dict["affinity_token_mask"] = (affinity_mask > 0).float()

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

        data = {
            "input_feature_dict": feat,
            "score": torch.tensor([score], dtype=torch.float32),
        }

        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0] if "msa" in feat else 1
        data.update(
            {
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                "N_msa": torch.tensor([N_msa]),
            }
        )
        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test AffInferenceDataset")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--n", type=int, default=2, help="Yield this many samples then stop.")
    args = parser.parse_args()

    dataset = AffInferenceDataset(csv_path=args.csv_path)
    for i, sample in enumerate(dataset):
        print(
            f"[{i}] prot_id={sample['prot_id']}  "
            f"N_token={int(sample['N_token'])}  "
            f"N_atom={int(sample['N_atom'])}  "
            f"N_msa={int(sample['N_msa'])}  "
            f"score={float(sample['score'])}"
        )
        if i + 1 >= args.n:
            break
