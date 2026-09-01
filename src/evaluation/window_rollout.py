from __future__ import annotations

from typing import Any

import torch


def model_mean(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output.get("mean", next(iter(output.values())))
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported model output type: {type(output).__name__}")
    return output


@torch.no_grad()
def rollout_trajectory(
    model: torch.nn.Module,
    x_static: torch.Tensor,
    y0: torch.Tensor,
    T: int,
    K: int,
    include_source: bool,
    use_prev: bool,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Predict eta frames 1..T-1 autoregressively from the given eta frame 0."""
    del device

    T = int(T)
    K = int(K)
    if T < 1:
        raise ValueError(f"T must be positive, got {T}")
    if K < 1:
        raise ValueError(f"K must be positive, got {K}")
    if x_static.ndim != 4 or x_static.shape[1] < 2:
        raise ValueError(
            "x_static must have shape [B,C,H,W] with bathymetry and source channels"
        )
    if y0.ndim != 3:
        raise ValueError(f"y0 must have shape [B,H,W], got {tuple(y0.shape)}")
    if T == 1:
        return y0[:, :0].reshape(y0.shape[0], 0, *y0.shape[1:])

    bathymetry = x_static[:, 0]
    source = x_static[:, 1]
    eta_t = y0
    eta_prev = y0
    predictions: list[torch.Tensor] = []
    produced = 0
    target_len = T - 1

    while produced < target_len:
        channels = [bathymetry, source] if include_source else [bathymetry]
        channels.append(eta_t)
        if use_prev:
            channels.append(eta_prev)
        window_input = torch.stack(channels, dim=1)
        window_prediction = model_mean(model(window_input))
        if window_prediction.ndim != 4:
            raise ValueError(
                "Windowed model output must have shape [B,K,H,W], got "
                f"{tuple(window_prediction.shape)}"
            )
        if window_prediction.shape[0] != y0.shape[0]:
            raise ValueError("Windowed model output batch size does not match the seed")
        if window_prediction.shape[2:] != y0.shape[1:]:
            raise ValueError("Windowed model output spatial shape does not match the seed")
        if window_prediction.shape[1] < 1:
            raise ValueError("Windowed model produced no future frames")

        predictions.append(window_prediction)
        chunk_size = int(window_prediction.shape[1])
        eta_prev = (
            window_prediction[:, -2]
            if chunk_size >= 2
            else window_prediction[:, -1]
        )
        eta_t = window_prediction[:, -1]
        produced += chunk_size

    return torch.cat(predictions, dim=1)[:, :target_len]
