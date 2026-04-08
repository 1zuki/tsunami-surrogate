from __future__ import annotations

from src.models.cnn import CNNForecaster
from src.models.convlstm import ConvLSTMForecaster
from src.models.fno import FNO2d
from src.models.unet import UNetForecaster


def build_model(config: dict):
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "fno")).lower()
    in_channels = int(model_cfg.get("in_channels", 2))
    out_channels = int(model_cfg.get("out_channels", config.get("simulation", {}).get("nt", 20)))
    use_grid = bool(model_cfg.get("use_grid", True))

    if name == "fno":
        return FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(model_cfg.get("hidden_channels", 64)),
            modes_x=int(model_cfg.get("modes_x", 12)),
            modes_y=int(model_cfg.get("modes_y", 12)),
            n_layers=int(model_cfg.get("n_layers", 4)),
            padding=int(model_cfg.get("padding", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_grid=use_grid,
        )
    if name == "unet":
        return UNetForecaster(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(model_cfg.get("base_channels", 32)),
            depth=int(model_cfg.get("depth", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_grid=use_grid,
        )
    if name == "cnn":
        return CNNForecaster(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(model_cfg.get("hidden_channels", 64)),
            n_blocks=int(model_cfg.get("n_blocks", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_grid=use_grid,
        )
    if name == "convlstm":
        return ConvLSTMForecaster(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(model_cfg.get("hidden_channels", 64)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_grid=use_grid,
        )
    raise ValueError(f"Unknown model name: {name}")
