from .fno2d import FNO2D
from .uncertainty import ProbabilisticFNO2D


def build_model(cfg):
    model_cfg = cfg.get('model', cfg)
    name = model_cfg.get('name', 'fno2d')
    if name == 'fno2d':
        return FNO2D(
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            modes1=model_cfg.get('modes1', 12),
            modes2=model_cfg.get('modes2', 12),
            width=model_cfg.get('width', 32),
            depth=model_cfg.get('depth', 4),
            padding=model_cfg.get('padding', 6),
            use_grid=model_cfg.get('use_grid', True),
        )
    if name == 'fno2d_probabilistic':
        return ProbabilisticFNO2D(
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            modes1=model_cfg.get('modes1', 12),
            modes2=model_cfg.get('modes2', 12),
            width=model_cfg.get('width', 32),
            depth=model_cfg.get('depth', 4),
            padding=model_cfg.get('padding', 6),
            use_grid=model_cfg.get('use_grid', True),
        )
    raise ValueError(f'Unknown model name: {name}')
