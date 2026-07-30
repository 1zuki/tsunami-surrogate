from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from src.utils.seed import make_torch_generator, make_worker_init_fn
from src.utils.hashing import sha256_file


PROCESSED_MANIFEST_SCHEMA_ID = "tsunami-surrogate.processed-dataset.v2"


def _load_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _validate_normalization_provenance(
    root: Path,
    manifest: Dict[str, Any],
    manifest_path: Path,
) -> None:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Processed v2 manifest is missing provenance: {manifest_path}")
    normalization = provenance.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError(
            f"Processed v2 manifest is missing normalization provenance: "
            f"{manifest_path}"
        )

    normalization_path = root / str(
        normalization.get("path", "../normalization_stats.json")
    )
    expected_hash = str(normalization.get("sha256", ""))
    if not normalization_path.is_file() or not expected_hash:
        raise ValueError(
            f"Processed v2 normalization artifact is incomplete: "
            f"{normalization_path}"
        )
    if sha256_file(normalization_path) != expected_hash:
        raise RuntimeError(
            f"Processed normalization hash mismatch: {normalization_path}"
        )


def _resolve_array_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    if resolved.is_dir():
        candidate = resolved / "eval_dataset.npz"
        if candidate.exists():
            return candidate

        npz_candidates = sorted(resolved.glob("*.npz"))
        if not npz_candidates:
            raise FileNotFoundError(f"No .npz found in directory: {resolved}")
        return npz_candidates[0]

    return resolved


def _validate_unsharded_v2_artifact(path: Path) -> Dict[str, Any] | None:
    manifest_path = path.parent / "eval_manifest.json"
    if not manifest_path.is_file():
        return None

    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_id") != PROCESSED_MANIFEST_SCHEMA_ID:
        return None
    if bool(manifest.get("sharded", False)):
        raise ValueError(
            f"Sharded processed manifest cannot describe flat dataset {path}"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(
            f"Processed v2 manifest is missing artifact hashes: {manifest_path}"
        )
    expected_hash = str(artifacts.get(path.name, ""))
    if not expected_hash:
        raise ValueError(
            f"Processed v2 manifest does not bind dataset artifact {path.name}: "
            f"{manifest_path}"
        )
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"Processed dataset hash mismatch: {path}")

    _validate_normalization_provenance(path.parent, manifest, manifest_path)
    return manifest


def _as_nchw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr[:, None, :, :].astype(np.float32)
    if arr.ndim == 4:
        return arr.astype(np.float32)

    raise ValueError(f"Expected [N,H,W] or [N,C,H,W], got {arr.shape}")


def _to_nchw_if_single(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr[:, None, :, :].astype(np.float32)
    if arr.ndim == 4:
        return arr.astype(np.float32)
    if arr.ndim == 5 and arr.shape[2] == 1:
        return arr[:, :, 0, :, :].astype(np.float32)

    return arr.astype(np.float32)


def save_npz(path: str | Path, x: np.ndarray, y: np.ndarray, metadata: Dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        metadata = {}

    np.savez_compressed(path, x=np.asarray(x), y=np.asarray(y), metadata=np.array([metadata], dtype=object))


@dataclass
class LoadedArrays:
    x: np.ndarray
    y: np.ndarray
    sample_id: np.ndarray
    source_id: np.ndarray
    source_type: np.ndarray
    bathymetry_type: np.ndarray
    source_strength: np.ndarray
    scenario_id: np.ndarray
    solver_name: np.ndarray


def _string_array(data: Any, n: int, default: str) -> np.ndarray:
    arr = np.asarray(data) if data is not None else np.asarray([], dtype=object)
    if arr.size == 0:
        return np.asarray([default] * n, dtype=object)

    arr = arr.reshape(-1)
    out = np.asarray([str(v) for v in arr], dtype=object)

    if out.shape[0] != n:
        raise ValueError(
            f"Metadata length mismatch: expected {n}, got {out.shape[0]}"
        )

    return out


def _float_array(data: Any, n: int, default: float = np.nan) -> np.ndarray:
    arr = np.asarray(data) if data is not None else np.asarray([], dtype=float)
    if arr.size == 0:
        return np.full((n,), float(default), dtype=np.float32)

    arr = arr.reshape(-1).astype(np.float32)
    if arr.shape[0] != n:
        raise ValueError(
            f"Numeric metadata length mismatch: expected {n}, got {arr.shape[0]}"
        )

    return arr


def _load_arrays(path: str | Path) -> LoadedArrays:
    path = _resolve_array_path(path)

    with np.load(path, allow_pickle=True) as data:
        if "x" in data and "y" in data:
            x = _as_nchw(data["x"])
            y = _to_nchw_if_single(data["y"])

        elif "inputs" in data and "targets" in data:
            x = _as_nchw(data["inputs"])
            y = _to_nchw_if_single(data["targets"])

        else:
            raise KeyError(f"Unsupported dataset keys in {path}. Expected x/y or inputs/targets.")

        n = x.shape[0]

        sample_id = _string_array(
            data["sample_id"] if "sample_id" in data else None,
            n,
            default="",
        )
        if not np.any(sample_id != ""):
            sample_id = np.asarray([f"sample_{i:06d}" for i in range(n)], dtype=object)

        source_type = _string_array(
            data["source_type"] if "source_type" in data else None,
            n,
            default="unknown",
        )
        source_id = _string_array(
            data["source_id"] if "source_id" in data else source_type,
            n,
            default="unknown",
        )
        bathymetry_type = _string_array(
            data["bathymetry_type"] if "bathymetry_type" in data else None,
            n,
            default="unknown",
        )
        source_strength = _float_array(
            data["source_strength"] if "source_strength" in data else None,
            n,
            default=np.nan,
        )
        scenario_id = _string_array(
            data["scenario_id"] if "scenario_id" in data else sample_id,
            n,
            default="",
        )
        solver_name = _string_array(
            data["solver_name"] if "solver_name" in data else None,
            n,
            default="unknown",
        )

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched sample count: x={x.shape[0]} y={y.shape[0]}")

    return LoadedArrays(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        sample_id=sample_id,
        source_id=source_id,
        source_type=source_type,
        bathymetry_type=bathymetry_type,
        source_strength=source_strength,
        scenario_id=scenario_id,
        solver_name=solver_name,
    )


class TsunamiDataset(Dataset):
    def __init__(self, path: str | Path):
        resolved_path = _resolve_array_path(path)
        manifest = _validate_unsharded_v2_artifact(resolved_path)
        arrays = _load_arrays(resolved_path)
        if manifest is not None:
            expected_inputs_shape = manifest.get("inputs_shape")
            expected_targets_shape = manifest.get("targets_shape")
            if isinstance(expected_inputs_shape, list) and list(arrays.x.shape) != [
                int(value) for value in expected_inputs_shape
            ]:
                raise ValueError(
                    f"Processed input-shape mismatch for {resolved_path}: "
                    f"manifest={expected_inputs_shape}, observed={list(arrays.x.shape)}"
                )
            if isinstance(expected_targets_shape, list) and list(arrays.y.shape) != [
                int(value) for value in expected_targets_shape
            ]:
                raise ValueError(
                    f"Processed target-shape mismatch for {resolved_path}: "
                    f"manifest={expected_targets_shape}, observed={list(arrays.y.shape)}"
                )
        self.x = torch.from_numpy(arrays.x)
        self.y = torch.from_numpy(arrays.y)
        self.sample_id = arrays.sample_id
        self.source_id = arrays.source_id
        self.source_type = arrays.source_type
        self.bathymetry_type = arrays.bathymetry_type
        self.source_strength = arrays.source_strength
        self.scenario_id = arrays.scenario_id
        self.solver_name = arrays.solver_name

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "sample_id": str(self.sample_id[idx]),
            "source_id": str(self.source_id[idx]),
            "source_type": str(self.source_type[idx]),
            "bathymetry_type": str(self.bathymetry_type[idx]),
            "source_strength": float(self.source_strength[idx]),
            "scenario_id": str(self.scenario_id[idx]),
            "solver_name": str(self.solver_name[idx]),
        }


class ShardedTsunamiDataset(Dataset):
    def __init__(self, path: str | Path, cache_size: int = 2):
        self.root = Path(path)
        self.manifest_path = self.root / "shards_manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)

        manifest = _load_json_object(self.manifest_path)

        self.shards: List[Dict[str, Any]] = list(manifest.get("shards", []))
        self.schema_id = str(manifest.get("schema_id", ""))
        self._validated_shards: set[int] = set()
        if self.schema_id == PROCESSED_MANIFEST_SCHEMA_ID:
            _validate_normalization_provenance(
                self.root,
                manifest,
                self.manifest_path,
            )
            for shard in self.shards:
                if not shard.get("file") or not shard.get("sha256"):
                    raise ValueError(
                        f"Processed v2 shard entry lacks file/hash: {self.manifest_path}"
                    )

        self.counts = [int(shard.get("num_samples", 0)) for shard in self.shards]
        self.cumulative: List[int] = []
        total = 0
        for count in self.counts:
            total += count
            self.cumulative.append(total)

        self.num_samples = int(manifest.get("num_samples", total))
        if self.num_samples != total:
            if self.schema_id == PROCESSED_MANIFEST_SCHEMA_ID:
                raise ValueError(
                    f"Processed v2 sample-count mismatch in {self.manifest_path}: "
                    f"manifest={self.num_samples}, shards={total}"
                )
            self.num_samples = total

        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, LoadedArrays] = OrderedDict()

    def __len__(self) -> int:
        return self.num_samples

    def shard_index_for_sample(self, idx: int) -> int:
        idx = int(idx)
        if idx < 0:
            idx += self.num_samples
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)
        return bisect.bisect_right(self.cumulative, idx)

    def _load_shard(self, shard_idx: int) -> LoadedArrays:
        cached = self._cache.get(shard_idx)
        if cached is not None:
            self._cache.move_to_end(shard_idx)
            return cached

        shard_file = self.shards[shard_idx].get("file")
        if not shard_file:
            raise KeyError(f"Shard {shard_idx} in {self.manifest_path} is missing a file path.")

        shard_path = self.root / str(shard_file)
        if self.schema_id == PROCESSED_MANIFEST_SCHEMA_ID and shard_idx not in self._validated_shards:
            expected_hash = str(self.shards[shard_idx]["sha256"])
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            if sha256_file(shard_path) != expected_hash:
                raise RuntimeError(f"Processed shard hash mismatch: {shard_path}")
            self._validated_shards.add(shard_idx)

        arrays = _load_arrays(shard_path)
        self._cache[shard_idx] = arrays
        self._cache.move_to_end(shard_idx)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = int(idx.item())
        idx = int(idx)
        if idx < 0:
            idx += self.num_samples
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)

        shard_idx = self.shard_index_for_sample(idx)
        shard_start = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = idx - shard_start
        arrays = self._load_shard(shard_idx)

        return {
            "x": torch.from_numpy(arrays.x[local_idx]),
            "y": torch.from_numpy(arrays.y[local_idx]),
            "sample_id": str(arrays.sample_id[local_idx]),
            "source_id": str(arrays.source_id[local_idx]),
            "source_type": str(arrays.source_type[local_idx]),
            "bathymetry_type": str(arrays.bathymetry_type[local_idx]),
            "source_strength": float(arrays.source_strength[local_idx]),
            "scenario_id": str(arrays.scenario_id[local_idx]),
            "solver_name": str(arrays.solver_name[local_idx]),
        }


class ShardedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        source = _sharded_dataset_with_parent_indices(dataset)
        if source is None:
            raise TypeError("ShardedBatchSampler requires a ShardedTsunamiDataset or a Subset wrapping one.")

        sharded_dataset, parent_indices = source
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        groups: List[List[int]] = [[] for _ in sharded_dataset.shards]
        for local_idx, parent_idx in enumerate(parent_indices):
            shard_idx = sharded_dataset.shard_index_for_sample(int(parent_idx))
            groups[shard_idx].append(local_idx)

        self.groups = [group for group in groups if group]
        if self.drop_last:
            self._num_batches = sum(len(group) // self.batch_size for group in self.groups)
        else:
            self._num_batches = sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        group_order = rng.permutation(len(self.groups)).tolist()
        for group_idx in group_order:
            group = self.groups[group_idx]
            sample_order = rng.permutation(len(group)).tolist()
            shuffled = [group[i] for i in sample_order]

            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        return int(self._num_batches)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self._epoch = epoch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": int(self._epoch),
            "seed": int(self.seed),
            "batch_size": int(self.batch_size),
            "drop_last": bool(self.drop_last),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("ShardedBatchSampler seed mismatch during resume")
        if int(state.get("batch_size", self.batch_size)) != self.batch_size:
            raise ValueError("ShardedBatchSampler batch-size mismatch during resume")
        if bool(state.get("drop_last", self.drop_last)) != self.drop_last:
            raise ValueError("ShardedBatchSampler drop-last mismatch during resume")
        self.set_epoch(int(state.get("epoch", 0)))


class WindowedShardBatchSampler(Sampler[List[int]]):
    """Shard-local batches for a WindowedTrajectoryDataset.

    Groups window indices by the shard their base sample lives in, shuffles
    within each shard, and emits batches that stay inside one shard. This keeps
    the underlying ShardedTsunamiDataset's small LRU cache hot (no per-item
    shard reloads), which is what makes shuffled windowed training feasible
    without exhausting host RAM or thrashing disk.
    """

    def __init__(self, dataset, batch_size: int, seed: int, drop_last: bool = False) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        base = getattr(dataset, "base", None)
        source = _sharded_dataset_with_parent_indices(base) if base is not None else None
        if source is None:
            raise TypeError("WindowedShardBatchSampler requires a WindowedTrajectoryDataset over a sharded base.")
        sharded_dataset, parent_indices = source

        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        wps = int(dataset.windows_per_sample)
        groups: List[List[int]] = [[] for _ in sharded_dataset.shards]
        for win_idx in range(len(dataset)):
            base_local = win_idx // wps
            parent_idx = int(parent_indices[base_local])
            shard_idx = sharded_dataset.shard_index_for_sample(parent_idx)
            groups[shard_idx].append(win_idx)

        self.groups = [g for g in groups if g]
        if self.drop_last:
            self._num_batches = sum(len(g) // self.batch_size for g in self.groups)
        else:
            self._num_batches = sum((len(g) + self.batch_size - 1) // self.batch_size for g in self.groups)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        group_order = rng.permutation(len(self.groups)).tolist()
        for group_idx in group_order:
            group = self.groups[group_idx]
            order = rng.permutation(len(group)).tolist()
            shuffled = [group[i] for i in order]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        return int(self._num_batches)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self._epoch = epoch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": int(self._epoch),
            "seed": int(self.seed),
            "batch_size": int(self.batch_size),
            "drop_last": bool(self.drop_last),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("WindowedShardBatchSampler seed mismatch during resume")
        if int(state.get("batch_size", self.batch_size)) != self.batch_size:
            raise ValueError(
                "WindowedShardBatchSampler batch-size mismatch during resume"
            )
        if bool(state.get("drop_last", self.drop_last)) != self.drop_last:
            raise ValueError(
                "WindowedShardBatchSampler drop-last mismatch during resume"
            )
        self.set_epoch(int(state.get("epoch", 0)))


def _split_indices(n: int, split_cfg: Dict[str, Any], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_type = str(split_cfg.get("type", "iid")).lower()

    if split_type != "iid":
        # fallback to deterministic IID-style split for unsupported modes
        split_type = "iid"

    train_ratio = float(split_cfg.get("train", 0.7))
    val_ratio = float(split_cfg.get("val", 0.15))
    test_ratio = float(split_cfg.get("test", 0.15))

    total = train_ratio + val_ratio + test_ratio

    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), max(n - 2, 1)) if n >= 3 else max(min(n_train, n), 0)
    n_val = min(max(n_val, 0), max(n - n_train - 1, 0)) if n >= 2 else 0
    n_test_start = n_train + n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_test_start]
    test_idx = perm[n_test_start:]

    if test_idx.size == 0 and val_idx.size > 0:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]

    if test_idx.size == 0 and train_idx.size > 0:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    return train_idx, val_idx, test_idx


class _InMemoryDataset(Dataset):
    """Flat in-memory dataset built from selected indices of a source dataset.

    Used for sample-scaling subsets of a sharded dataset. Materializing the
    chosen samples lets the standard shuffle loader apply a constant batch size,
    instead of the shard-local batch sampler whose effective batch size scales
    with how the subset happens to scatter across shards -- which would confound
    a sample-scaling sweep by training small-N points with tiny noisy batches.
    """

    def __init__(self, source: Dataset, indices: List[int]):
        # Load in ascending index order so each shard is read once (shard index
        # ranges are contiguous), avoiding LRU-cache thrashing during the build.
        items = [source[int(i)] for i in sorted(int(i) for i in indices)]
        self.x = torch.stack([torch.as_tensor(it["x"]).float() for it in items])
        self.y = torch.stack([torch.as_tensor(it["y"]).float() for it in items])
        self.sample_id = [str(it.get("sample_id", "")) for it in items]
        self.source_id = [str(it.get("source_id", "")) for it in items]
        self.source_type = [str(it.get("source_type", "")) for it in items]
        self.bathymetry_type = [str(it.get("bathymetry_type", "")) for it in items]
        self.source_strength = [float(it.get("source_strength", 0.0)) for it in items]
        self.scenario_id = [str(it.get("scenario_id", "")) for it in items]
        self.solver_name = [str(it.get("solver_name", "")) for it in items]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "sample_id": self.sample_id[idx],
            "source_id": self.source_id[idx],
            "source_type": self.source_type[idx],
            "bathymetry_type": self.bathymetry_type[idx],
            "source_strength": self.source_strength[idx],
            "scenario_id": self.scenario_id[idx],
            "solver_name": self.solver_name[idx],
        }


def _parse_optional_sample_count(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None

    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be a positive integer or null, got {value!r}.")
    return count


def _sample_limit_for_split(data_cfg: Dict[str, Any], split_name: str) -> int | None:
    aliases = {
        "train": ("train_samples", "n_train_samples", "max_train_samples", "n_samples"),
        "val": ("val_samples", "n_val_samples", "max_val_samples"),
        "test": ("test_samples", "n_test_samples", "max_test_samples"),
    }

    for key in aliases.get(split_name, (f"{split_name}_samples",)):
        if key in data_cfg:
            return _parse_optional_sample_count(data_cfg.get(key), f"data.{key}")
    return None


def _limit_dataset(dataset: Dataset, n_samples: int | None, seed: int) -> Dataset:
    if n_samples is None or n_samples >= len(dataset):
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:n_samples].tolist()

    # For sharded sources, materialize the subset into a flat in-memory dataset
    # so the standard shuffle loader applies a constant batch size. A Subset over
    # a ShardedTsunamiDataset would instead route through the shard-local batch
    # sampler, whose effective batch size scales with the subset size -- a
    # confound for sample-scaling sweeps. Safe to materialize here: subsets are
    # small (<= a few thousand) relative to the full pool.
    if _sharded_dataset_with_parent_indices(dataset) is not None:
        return _InMemoryDataset(dataset, indices)

    return Subset(dataset, indices)


def _prepare_split_dataset(data_cfg: Dict[str, Any], split_name: str, dataset: Dataset, seed: int) -> Dataset:
    limited = _limit_dataset(
        dataset,
        _sample_limit_for_split(data_cfg, split_name),
        seed,
    )
    if bool(data_cfg.get("windowed", False)):
        from .window_dataset import WindowedTrajectoryDataset

        return WindowedTrajectoryDataset(
            limited,
            K=int(data_cfg.get("window_K", 5)),
            prev=bool(data_cfg.get("window_prev", True)),
            include_source=bool(data_cfg.get("window_include_source", True)),
        )
    return limited



def _sharded_dataset_with_parent_indices(dataset: Dataset) -> Tuple[ShardedTsunamiDataset, List[int]] | None:
    if isinstance(dataset, ShardedTsunamiDataset):
        return dataset, list(range(len(dataset)))

    if isinstance(dataset, Subset) and isinstance(dataset.dataset, ShardedTsunamiDataset):
        return dataset.dataset, [int(idx) for idx in dataset.indices]

    return None


def _is_windowed_over_sharded(dataset: Dataset) -> bool:
    base = getattr(dataset, "base", None)
    if base is None or not hasattr(dataset, "windows_per_sample"):
        return False
    return _sharded_dataset_with_parent_indices(base) is not None


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    if shuffle and _sharded_dataset_with_parent_indices(dataset) is not None:
        batch_sampler = ShardedBatchSampler(dataset, batch_size=batch_size, seed=seed)
        print(
            "[data] using shard-aware batch sampler "
            f"samples={len(dataset)} batches={len(batch_sampler)} batch_size={batch_size}"
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            worker_init_fn=make_worker_init_fn(seed),
        )

    if shuffle and _is_windowed_over_sharded(dataset):
        batch_sampler = WindowedShardBatchSampler(dataset, batch_size=batch_size, seed=seed)
        print(
            "[data] using windowed shard-local batch sampler "
            f"windows={len(dataset)} batches={len(batch_sampler)} batch_size={batch_size}"
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            worker_init_fn=make_worker_init_fn(seed),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=make_worker_init_fn(seed),
        generator=make_torch_generator(seed),
    )


def _has_sharded_dataset(path: str | Path) -> bool:
    p = Path(path)
    manifest_path = p / "shards_manifest.json"
    if not manifest_path.is_file():
        return False

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return False

    return int(manifest.get("num_samples", 0)) > 0


def _resolve_dataset_path(path: str | Path) -> Path:
    p = Path(path)
    if p.name == "eval_dataset.npz" and _has_sharded_dataset(p.parent):
        return p.parent
    if p.exists():
        return p
    return p


def _has_npz_dataset(path: str | Path) -> bool:
    p = _resolve_dataset_path(path)
    if not p.exists():
        return False
    if p.is_file():
        return p.suffix == ".npz"
    if _has_sharded_dataset(p):
        return True
    if (p / "eval_dataset.npz").exists():
        return True
    
    return any(p.glob("*.npz"))


def _make_dataset(path: str | Path) -> Dataset:
    p = _resolve_dataset_path(path)
    if p.is_dir() and _has_sharded_dataset(p):
        return ShardedTsunamiDataset(p)
    return TsunamiDataset(p)


def create_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    data_cfg = cfg.get("data", cfg.get("dataset", {}))

    if not data_cfg:
        raise KeyError("Config must contain `data` or `dataset` section.")

    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", data_cfg.get("seed", 42)))

    loaders: Dict[str, DataLoader] = {}

    # Preferred mode: use explicit pre-split datasets if provided.
    split_paths = {
        "train": data_cfg.get("train_path"),
        "val": data_cfg.get("val_path"),
        "test": data_cfg.get("test_path"),
    }
    if any(v is not None for v in split_paths.values()):
        for split_name, split_path in split_paths.items():
            if split_path is None:
                continue
            if not _has_npz_dataset(split_path):
                continue
            split_seed = seed + {"train": 0, "val": 1, "test": 2}[split_name]
            split_dataset = _prepare_split_dataset(data_cfg, split_name, _make_dataset(split_path), split_seed)
            loaders[split_name] = _make_loader(
                split_dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                seed=split_seed,
                num_workers=num_workers,
            )
        if not loaders:
            raise ValueError("Explicit split-path mode was requested, but no split dataset could be loaded.")
        return loaders

    path = data_cfg.get("path")
    if path is None:
        raise KeyError("data.path is required when explicit train_path/val_path/test_path are not provided.")

    # Auto-detect common pre-split folder layout:
    # data/processed/{train,val,test}/eval_dataset.npz
    root = Path(path)
    if root.is_dir():
        pre_split = {
            "train": root / "train",
            "val": root / "val",
            "test": root / "test",
        }
        if any(p.exists() for p in pre_split.values()):
            for split_name, split_path in pre_split.items():
                if not split_path.exists():
                    continue
                if not _has_npz_dataset(split_path):
                    continue
                split_seed = seed + {"train": 0, "val": 1, "test": 2}[split_name]
                split_dataset = _prepare_split_dataset(data_cfg, split_name, _make_dataset(split_path), split_seed)
                loaders[split_name] = _make_loader(
                    split_dataset,
                    batch_size=batch_size,
                    shuffle=(split_name == "train"),
                    seed=split_seed,
                    num_workers=num_workers,
                )
            if loaders:
                return loaders

    # Backward-compatible mode: one dataset path + random split.
    dataset = _make_dataset(path)
    split_cfg = data_cfg.get("split", {"type": "iid"})
    train_idx, val_idx, test_idx = _split_indices(len(dataset), split_cfg, seed)

    if train_idx.size > 0:
        loaders["train"] = _make_loader(
            _prepare_split_dataset(data_cfg, "train", Subset(dataset, train_idx.tolist()), seed),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
        )
    if val_idx.size > 0:
        loaders["val"] = _make_loader(
            _prepare_split_dataset(data_cfg, "val", Subset(dataset, val_idx.tolist()), seed + 1),
            batch_size=batch_size,
            shuffle=False,
            seed=seed + 1,
            num_workers=num_workers,
        )
    if test_idx.size > 0:
        loaders["test"] = _make_loader(
            _prepare_split_dataset(data_cfg, "test", Subset(dataset, test_idx.tolist()), seed + 2),
            batch_size=batch_size,
            shuffle=False,
            seed=seed + 2,
            num_workers=num_workers,
        )
    return loaders
