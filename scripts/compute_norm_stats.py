"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import math

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    print("DEBUG: creating torch dataset...", flush=True)
    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
    )
    print(f"DEBUG: dataset created, len={len(dataset)}", flush=True)

    print("DEBUG: creating TransformedDataset...", flush=True)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX
            # and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    print("DEBUG: TransformedDataset created", flush=True)

    if len(dataset) == 0:
        raise ValueError("Dataset length is 0, cannot compute norm stats.")

    if max_frames is not None and max_frames < len(dataset):
        num_frames = max_frames
        shuffle = True
    else:
        num_frames = len(dataset)
        shuffle = False

    num_batches = max(1, math.ceil(num_frames / batch_size))

    print(
        f"DEBUG: batch_size={batch_size}, num_frames={num_frames}, "
        f"num_batches={num_batches}, shuffle={shuffle}",
        flush=True,
    )

    print(f"DEBUG: creating TorchDataLoader (num_workers={num_workers})...", flush=True)
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    print("DEBUG: TorchDataLoader created", flush=True)

    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    print("DEBUG: creating RLDS dataset...", flush=True)
    dataset = _data_loader.create_rlds_dataset(
        data_config,
        action_horizon,
        batch_size,
        shuffle=False,
    )
    print(f"DEBUG: RLDS dataset created, len={len(dataset)}", flush=True)

    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )

    if len(dataset) == 0:
        raise ValueError("RLDS dataset length is 0, cannot compute norm stats.")

    if max_frames is not None and max_frames < len(dataset):
        num_frames = max_frames
    else:
        num_frames = len(dataset)

    num_batches = max(1, math.ceil(num_frames / batch_size))

    print(
        f"DEBUG: batch_size={batch_size}, num_frames={num_frames}, "
        f"num_batches={num_batches}",
        flush=True,
    )

    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    print("DEBUG: RLDSDataLoader created", flush=True)

    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    print(f"DEBUG: config_name={config_name}", flush=True)
    print(f"DEBUG: repo_id={data_config.repo_id}", flush=True)
    print(f"DEBUG: assets_dirs={config.assets_dirs}", flush=True)
    print(f"DEBUG: batch_size={config.batch_size}", flush=True)
    print(f"DEBUG: action_horizon={config.model.action_horizon}", flush=True)
    print(f"DEBUG: num_workers={config.num_workers}", flush=True)

    if data_config.repo_id is None:
        raise ValueError("data_config.repo_id is None, cannot decide output path.")

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            max_frames,
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )

    print(f"DEBUG: dataloader created, num_batches={num_batches}", flush=True)

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    seen_batches = 0

    for i, batch in enumerate(
        tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats")
    ):
        seen_batches += 1

        batch_keys = list(batch.keys())
        print(f"DEBUG: batch {i}, keys={batch_keys}", flush=True)

        for key in keys:
            if key not in batch:
                raise KeyError(
                    f"Expected key '{key}' in batch, but got keys={batch_keys}"
                )

            arr = np.asarray(batch[key])
            print(f"DEBUG: batch {i}, {key}_shape={arr.shape}", flush=True)
            stats[key].update(arr)

    if seen_batches == 0:
        raise RuntimeError(
            "Dataloader produced 0 batches. "
            "Please check batch_size, dataset length, and TorchDataLoader behavior."
        )

    print(f"DEBUG: finished stats computation, seen_batches={seen_batches}", flush=True)

    norm_stats = {key: running_stats.get_statistics() for key, running_stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Writing stats to: {output_path}", flush=True)
    normalize.save(output_path, norm_stats)
    print(f"Successfully saved norm stats to: {output_path / 'norm_stats.json'}", flush=True)


if __name__ == "__main__":
    tyro.cli(main)