"""Attach copied ColabFold/MMseqs2 MSA files to an inference CSV.

The input/index CSV must contain ``prot_id`` and the three hash columns
``h_msa_hash``, ``l_msa_hash`` and ``ag_msa_hash``.  The source MSA directory
is expected to contain the corresponding ``msa.files.H``, ``msa.files.L`` and
``msa.files.ag`` folders.  No machine-specific paths are embedded in this
script; all locations are supplied on the command line.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


CHAINS = [
    ("heavy_msa_path", "h_msa_hash", "msa.files.H", "H"),
    ("light_msa_path", "l_msa_hash", "msa.files.L", "L"),
    ("antigen_msa_path", "ag_msa_hash", "msa.files.ag", "ag"),
]


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or (
        isinstance(value, str) and not value.strip()
    )


def _path_for_csv(path: Path, style: str, relative_to: Path) -> str:
    if style == "absolute":
        return str(path.resolve())
    return os.path.relpath(path.resolve(), relative_to.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy per-chain MSAs and add *_msa_path columns to a CSV."
    )
    parser.add_argument("--target-csv", required=True, help="CSV whose rows will be evaluated.")
    parser.add_argument(
        "--index-csv",
        required=True,
        help="CSV mapping prot_id to h_msa_hash/l_msa_hash/ag_msa_hash.",
    )
    parser.add_argument(
        "--source-msa-dir",
        required=True,
        help="Directory containing msa.files.H, msa.files.L and msa.files.ag.",
    )
    parser.add_argument(
        "--output-msa-dir",
        required=True,
        help="Directory where the referenced .a3m files will be copied.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV; defaults to overwriting --target-csv.",
    )
    parser.add_argument(
        "--path-style",
        choices=("absolute", "relative"),
        default="absolute",
        help="How paths are written into the CSV (default: absolute).",
    )
    parser.add_argument(
        "--relative-to",
        default=None,
        help="Base directory for relative paths (default: output CSV directory).",
    )
    args = parser.parse_args()

    target_csv = Path(args.target_csv).resolve()
    index_csv = Path(args.index_csv).resolve()
    source_root = Path(args.source_msa_dir).resolve()
    output_root = Path(args.output_msa_dir).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else target_csv
    relative_to = Path(args.relative_to).resolve() if args.relative_to else output_csv.parent

    target = pd.read_csv(target_csv)
    index = pd.read_csv(index_csv)
    required_index = {"prot_id", *(hash_col for _, hash_col, _, _ in CHAINS)}
    missing = sorted(required_index - set(index.columns))
    if missing:
        raise ValueError(f"Index CSV is missing columns: {missing}")
    if "prot_id" not in target.columns:
        raise ValueError("Target CSV must contain a prot_id column.")

    lookups = {
        hash_col: dict(zip(index["prot_id"].astype(str), index[hash_col]))
        for _, hash_col, _, _ in CHAINS
    }
    for _, _, _, destination_subdir in CHAINS:
        (output_root / destination_subdir).mkdir(parents=True, exist_ok=True)

    for output_column, hash_column, source_subdir, destination_subdir in CHAINS:
        values = []
        for prot_id in target["prot_id"].astype(str):
            hash_value = lookups[hash_column].get(prot_id)
            if _is_missing(hash_value):
                values.append("")
                continue
            filename = f"{hash_value}.a3m"
            source = source_root / source_subdir / filename
            destination = output_root / destination_subdir / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing MSA for {prot_id} ({output_column}): {source}")
            if not destination.is_file():
                shutil.copy2(source, destination)
            values.append(_path_for_csv(destination, args.path_style, relative_to))
        target[output_column] = values

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(output_csv, index=False)
    print(f"Wrote {len(target)} rows to {output_csv}")
    print(f"MSAs copied into {output_root}")


if __name__ == "__main__":
    main()
