# FoldAbS inference

This directory contains the inference implementation and released head
weights for FoldAbS.  FoldAbS uses a frozen Protenix v1 backbone and adds
task-specific heads for antibody--antigen specificity, continuous affinity,
and developability.

The repository intentionally does **not** include the Protenix backbone or
the ESM-2 650M weights.  Install Protenix from the
[official Protenix repository](https://github.com/bytedance/Protenix), use its
v1 model implementation, and obtain the ESM-2 weights from the upstream ESM
release.

## Requirements

After cloning, fetch the Git LFS objects before running inference:

```bash
git lfs install
git lfs pull
```

- A pinned Protenix v1 checkout, available as `FoldAbS/Protenix/` (or exposed
  through `PYTHONPATH`), including its `configs` and `protenix` packages.
- PyTorch with a CUDA-capable GPU for practical inference.
- The Protenix v1 backbone checkpoint
  `protenix_base_default_v1.0.0.pt`.
- The ESM-2 650M checkpoint
  `esm2_t33_650M_UR50D.pt` for developability.
- Python packages used by the upstream Protenix environment, plus `pandas`,
  `PyYAML`, `numpy`, and `scipy` (the latter is only needed for optional
  metric printing).

The paths to the backbone and ESM weights are configured in:

- `configs/eval_spec.yml`
- `configs/eval_affinity.yml`
- `configs/eval_developability.yml`

Relative paths are resolved against the checkout, while absolute paths may be
used for externally mounted weights.

The released repository does not contain the backbone or ESM weights.  Edit
the relevant YAML file so `checkpoint_path` and (for developability)
`esm_module.model_dir` point to the downloaded files.

## Checkpoints

The `checkpoints/` directory contains state-only task weights:

| File | Contents |
|---|---|
| `spec_head.pt` | specificity head |
| `aff.pt`, `ske.pt`, `her2.pt` | affinity head + regression head |
| `aggre.pt`, `expre.pt`, `immun.pt`, `ther.pt` | developability head |

The developability release contains four v5 tasks.  Solubility is not part of
this release.  The checkpoints do not contain the Protenix backbone, ESM
weights, training data, or training configuration.

## MSA preparation

MSA search is deliberately outside this repository.  Before inference, run
ColabFold/MMseqs2 (or another compatible search pipeline) for every unique
heavy chain, light chain, and antigen sequence.  Save the resulting `.a3m`
files and add their paths to the input CSV.  The inference dataloaders do not
run MSA search and will fail fast when a required path is missing.

Required specificity/affinity columns:

```text
prot_id,heavy_sequence,light_sequence,antigen_sequence,
heavy_msa_path,light_msa_path,antigen_msa_path
```

`label` is optional for specificity; `score` is optional for affinity and is
used only for reporting metrics.

Required developability columns:

```text
sample_id,task,heavy,light,heavy_msa_path,light_msa_path
```

`target` is optional.  An `msa_id` column alone is not sufficient; it must be
resolved to an actual `.a3m` path first.

The checked-in ACVR2B example is already prepared with portable MSA paths.
For another dataset, use `python data/example/add_msa_paths.py --help` to
attach MSA files from a hash index, or create the required path columns
directly after your ColabFold search.  The helper has no hard-coded machine
paths.

## Inference

Run from the `FoldAbS/` directory.  The rollout values in the YAML file are
used by default; `--N_step` and `--N_sample` override them only when supplied.

Specificity:

```bash
python scripts/eval_spec.py \
  --config configs/eval_spec.yml \
  --spec_ckpt checkpoints/spec_head.pt \
  --csv_path data/example/acvr2b.csv \
  --out_csv results/specificity_predictions.csv
```

The checked-in ACVR2B CSV already contains portable paths to the checked-in
example MSAs, so it can be used directly after the backbone path is configured.

Affinity:

```bash
python scripts/eval_affinity.py \
  --config configs/eval_affinity.yml \
  --foldabs_ckpt checkpoints/aff.pt \
  --csv_path data/affinity/AFF/test_with_msa.csv \
  --out_csv results/affinity_predictions.csv
```

Use `ske.pt` or `her2.pt` for the corresponding affinity test set.  The
specificity CLI also accepts the legacy flag `--affinity_ckpt` for backward
compatibility.

Developability (one task per run):

```bash
python scripts/eval_developability.py \
  --config configs/eval_developability.yml \
  --head_ckpt checkpoints/aggre.pt \
  --task aggregation \
  --csv_path data/developability/test_with_msa.csv \
  --out_csv results/dev_aggregation.csv
```

Use `expre.pt`, `immun.pt`, or `ther.pt` and the matching `--task` for the
other three properties.  Add `--norm_stats` when raw-scale developability
values are required.

All three entry points support `torchrun` DDP.  For exact reproduction, keep
the Protenix commit, backbone/ESM checkpoint versions, YAML configuration,
MSA files, seed, and rollout overrides fixed.

## Repository hygiene

TAIJI submission files, runtime logs, generated predictions, Python caches,
and generated MSA files are ignored by `.gitignore`.  Do not add credentials,
Ceph mount commands, or internal filesystem paths to a public checkout.

## License and data

FoldAbS code and released task heads are provided under the MIT License in
`LICENSE`.  Protenix is an external dependency with its own Apache-2.0
license.  Verify that redistributed benchmark sequences, labels, MSA files,
and external model weights are covered by their respective licenses.
