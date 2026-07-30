from .fno2d import FNO2D
from .ffno2d import FFNO2D
from .ufno2d import UFNO2D
from .wno2d import WNO2D
from .uncertainty import ProbabilisticFNO2D
from .cnn import CNNBaseline
from .unet import UNetSmall
from .convlstm import ConvLSTMBaseline
from .signature import model_config_signature


def _tag_model(model, cfg):
    model._tsunami_model_config_signature = model_config_signature(cfg)
    model._tsunami_runtime_config = cfg
    return model


def build_model(cfg):
    model_cfg = cfg.get('model', cfg)
    name = model_cfg.get('name', 'fno2d')

    if name == 'fno2d':
        return _tag_model(FNO2D(
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            modes1=model_cfg.get('modes1', 12),
            modes2=model_cfg.get('modes2', 12),
            width=model_cfg.get('width', 32),
            depth=model_cfg.get('depth', 4),
            padding=model_cfg.get('padding', 6),
            use_grid=model_cfg.get('use_grid', True),
        ), cfg)
    if name == 'ffno2d':
        return _tag_model(FFNO2D(
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            modes1=model_cfg.get('modes1', 12),
            modes2=model_cfg.get('modes2', 12),
            width=model_cfg.get('width', 32),
            depth=model_cfg.get('depth', 4),
            padding=model_cfg.get('padding', 6),
            use_grid=model_cfg.get('use_grid', True),
        ), cfg)
    if name == 'wno2d':
        return _tag_model(WNO2D(
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            width=model_cfg.get('width', 32),
            depth=model_cfg.get('depth', 4),
            padding=model_cfg.get('padding', 6),
            use_grid=model_cfg.get('use_grid', True),
            wavelet_kernel_size=model_cfg.get('wavelet_kernel_size', 3),
        ), cfg)
    if name == 'ufno2d':
        return _tag_model(UFNO2D(
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            modes1=model_cfg.get('modes1', 12),
            modes2=model_cfg.get('modes2', 12),
            width=model_cfg.get('width', 32),
            depth=model_cfg.get('depth', 4),
            padding=model_cfg.get('padding', 6),
            use_grid=model_cfg.get('use_grid', True),
        ), cfg)
    if name == 'fno2d_probabilistic':
        return _tag_model(ProbabilisticFNO2D(
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            modes1=model_cfg.get('modes1', 12),
            modes2=model_cfg.get('modes2', 12),
            width=model_cfg.get('width', 32),
            depth=model_cfg.get('depth', 4),
            padding=model_cfg.get('padding', 6),
            use_grid=model_cfg.get('use_grid', True),
        ), cfg)
    if name == 'cnn':
        return _tag_model(CNNBaseline(
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            width=model_cfg.get('width', 32),
        ), cfg)
    if name == 'unet':
        return _tag_model(UNetSmall(
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            width=model_cfg.get('width', 32),
        ), cfg)
    if name == 'convlstm':
        return _tag_model(ConvLSTMBaseline(
            in_channels=model_cfg.get('in_channels', 3),
            out_channels=model_cfg.get('out_channels', 1),
            hidden_channels=model_cfg.get('hidden_channels', model_cfg.get('width', 48)),
            num_layers=model_cfg.get('num_layers', 2),
            kernel_size=model_cfg.get('kernel_size', 3),
            context_channels=model_cfg.get('context_channels', None),
            use_feedback=model_cfg.get('use_feedback', True),
        ), cfg)

    raise ValueError(f'Unknown model name: {name}')
