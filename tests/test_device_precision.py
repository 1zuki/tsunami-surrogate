import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.device import hardware_info, resolve_device
from src.utils.precision import configure_torch_precision, parse_optional_bool, tf32_backend_state


def test_resolve_device_cpu():
    dev = resolve_device("cpu")
    assert dev.type == "cpu"


def test_resolve_device_auto():
    dev = resolve_device("auto")
    assert dev.type in {"cpu", "cuda"}


def test_resolve_device_cuda_strict():
    if torch.cuda.is_available():
        dev = resolve_device("cuda")
        assert dev.type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_parse_optional_bool_variants():
    assert parse_optional_bool("true") is True
    assert parse_optional_bool("False") is False
    assert parse_optional_bool(True) is True
    assert parse_optional_bool(None) is None
    
    with pytest.raises(ValueError):
        parse_optional_bool("maybe")


def test_configure_precision_cpu_fp32():
    cfg = configure_torch_precision(torch.device("cpu"), precision="fp32", allow_tf32=False)
    assert cfg.name == "fp32"
    assert cfg.model_dtype == torch.float32
    assert cfg.input_dtype == torch.float32


def test_configure_precision_only_fp32_supported():
    with pytest.raises(ValueError):
        configure_torch_precision(torch.device("cpu"), precision="fp16")
    with pytest.raises(ValueError):
        configure_torch_precision(torch.device("cpu"), precision="bf16")
    with pytest.raises(ValueError):
        configure_torch_precision(torch.device("cpu"), precision="fp64")


def test_tf32_backend_state_cpu():
    matmul, cudnn = tf32_backend_state(torch.device("cpu"))
    assert matmul is None
    assert cudnn is None


def test_hardware_info_contains_expected_keys():
    info = hardware_info(torch.device("cpu"))
    for key in ("torch_version", "cpu", "selected_device", "cuda_available"):
        assert key in info
