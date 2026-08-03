"""DataLoader factory for ``DevInferenceDataset`` (inference-only).

Parallel to ``spec_inference_dataloader.py`` / ``aff_inference_dataloader.py``.
"""

from typing import Optional

from torch.utils.data import DataLoader

import yaml

from protenix.utils.torch_utils import collate_fn_first

from .dev_inference_dataset import DevInferenceDataset, TASK_NAMES


def get_dev_inference_dataloader(
    csv_path: str,
    task: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    dataset = DevInferenceDataset(csv_path=csv_path, task=task)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_first,
        num_workers=num_workers,
    )


def _summarize_batch(batch) -> None:
    print("Batch summary:")
    keys = sorted(batch.keys())
    print(f"Batch keys ({len(keys)}): {keys}")
    for key in keys:
        val = batch[key]
        if hasattr(val, "shape"):
            print(f"  {key}: shape={tuple(val.shape)}, dtype={getattr(val, 'dtype', None)}")
        else:
            print(f"  {key}: type={type(val)}")
    print("input_feature_dict summary:")
    keys = sorted(batch["input_feature_dict"].keys())
    print(f"input feature dict Keys ({len(keys)}): {keys}")
    for key in keys:
        val = batch["input_feature_dict"][key]
        if hasattr(val, "shape"):
            print(f"  {key}: shape={tuple(val.shape)}, dtype={getattr(val, 'dtype', None)}")
        else:
            print(f"  {key}: type={type(val)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Debug developability inference dataloader")
    parser.add_argument("--config", default=None,
                        help="Optional YAML; uses data.csv_path / data.task if provided.")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--task", default=None, choices=TASK_NAMES)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=1)
    args = parser.parse_args()

    csv_path = args.csv_path
    task = args.task
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        csv_path = csv_path or data_cfg.get("csv_path")
        task = task or data_cfg.get("task")

    if not csv_path:
        raise ValueError("csv_path is required (via --csv_path or --config data.csv_path).")

    dataloader = get_dev_inference_dataloader(
        csv_path=csv_path,
        task=task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}")
        _summarize_batch(batch)
        if "sample_id" in batch:
            print(f"  sample_id: {batch['sample_id']}  task: {batch.get('task')}")
        if i + 1 >= args.num_batches:
            break


if __name__ == "__main__":
    main()
