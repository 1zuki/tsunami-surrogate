from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def _loader_state(split: str, loader: Any) -> str:
    if loader is None:
        return f"{split}:missing"
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return f"{split}:dataset=unknown"
    try:
        n = int(len(ds))
    except Exception:
        return f"{split}:dataset=len-error"
    return f"{split}:dataset={n}"


def _first_batch(loaders: Dict[str, Any], preferred_splits: Iterable[str]) -> tuple[str, Dict[str, Any]]:
    diagnostics = []
    for split in preferred_splits:
        loader = loaders.get(split)
        diagnostics.append(_loader_state(split, loader))
        if loader is None:
            continue
        try:
            batch = next(iter(loader))
        except StopIteration:
            diagnostics.append(f"{split}:empty-iterator")
            continue
        except Exception as e:
            diagnostics.append(f"{split}:iter-error={type(e).__name__}")
            continue
        return split, batch
    diag_text = ", ".join(diagnostics) if diagnostics else "no preferred splits were provided"
    raise ValueError(
        "Could not read any batch from loaders to validate model I/O shapes. "
        f"Loader diagnostics: {diag_text}"
    )


def validate_model_io_channels(
    cfg: Dict[str, Any],
    loaders: Dict[str, Any],
    preferred_splits: Iterable[str] = ("train", "val", "test"),
) -> None:
    model_cfg = cfg.get("model", cfg)
    expected_in = int(model_cfg.get("in_channels", 0))
    expected_out = int(model_cfg.get("out_channels", 0))

    split, batch = _first_batch(loaders, preferred_splits)
    if "x" not in batch or "y" not in batch:
        raise KeyError(f"Batch from split '{split}' must contain 'x' and 'y' keys.")

    x = batch["x"]
    y = batch["y"]
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError(f"Expected x and y with at least 2 dims, got x={tuple(x.shape)} y={tuple(y.shape)}")

    actual_in = int(x.shape[1])
    actual_out = int(y.shape[1])

    if expected_in and expected_in != actual_in:
        raise ValueError(
            f"model.in_channels ({expected_in}) does not match dataset x channels ({actual_in}) on split '{split}'. "
            "If this is multifidelity preprocessing, check input.use_solver_id and update model.in_channels accordingly."
        )

    if expected_out and expected_out != actual_out:
        raise ValueError(
            f"model.out_channels ({expected_out}) does not match dataset y channels ({actual_out}) on split '{split}'. "
            "Check preprocess target.forecast_steps/variable and model.out_channels alignment."
        )
