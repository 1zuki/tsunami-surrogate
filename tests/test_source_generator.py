from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data_gen.generate_sources import SourceGenerator


def _write_config(
    tmp_path: Path,
    *,
    resolution: int,
    rough: dict[str, object],
    noise_enabled: bool = True,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "nx": resolution,
        "ny": resolution,
        "seed": None,
        "source_type": ["rough"],
        "gaussian": {
            "enabled": False,
            "amp_range": [0.1, 0.2],
            "sigma_range": [0.02, 0.04],
            "num_range": [1, 1],
        },
        "multi": {"enabled": False, "num_sources": [1, 1]},
        "okada": {
            "enabled": False,
            "length_range": [0.1, 0.2],
            "width_range": [0.05, 0.1],
            "slip_range": [0.1, 0.2],
            "angle_range": [0.0, 1.0],
            "dip_range": [0.0, 0.1],
            "depth_range": [0.0, 0.1],
            "smoothing_sigma": [0.1, 0.2],
        },
        "dipole": {
            "enabled": False,
            "amp_range": [0.1, 0.2],
            "sigma_range": [0.02, 0.04],
            "separation_range": [0.05, 0.1],
            "angle_range": [0.0, 1.0],
        },
        "fault": {
            "enabled": False,
            "amp_range": [0.1, 0.2],
            "length_range": [0.2, 0.3],
            "width_range": [0.02, 0.04],
            "angle_range": [0.0, 1.0],
            "smoothing_sigma": [0.01, 0.02],
        },
        "rough": {"enabled": True, **rough},
        "noise": {
            "enabled": noise_enabled,
            "scale_range": [0.5, 0.5],
            "smoothing_sigma": [1.0, 1.0],
        },
        "normalization": {
            "mode": "none",
            "clip_output": False,
            "height_scale": [-1.0, 1.0],
        },
    }
    path = tmp_path / f"source_{resolution}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_correlated_rough_is_deterministic_and_rms_controlled(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path,
        resolution=64,
        rough={
            "model": "correlated_grf",
            "amp_range": [0.1, 0.2],
            "smoothing_sigma": [1.0, 2.0],
            "correlation_length_range": [0.10, 0.10],
            "rms_range": [0.08, 0.08],
            "fine_amplitude_fraction": 0.0,
            "add_global_noise": False,
        },
    )
    first = SourceGenerator(str(path))
    first.rng = np.random.default_rng(1234)
    second = SourceGenerator(str(path))
    second.rng = np.random.default_rng(1234)

    first_field, first_type = first.generate()
    second_field, second_type = second.generate()

    assert first_type == second_type == "rough"
    np.testing.assert_array_equal(first_field, second_field)
    assert float(np.mean(first_field)) == pytest.approx(0.0, abs=1.0e-12)
    assert float(np.sqrt(np.mean(first_field * first_field))) == pytest.approx(
        0.08, abs=1.0e-12
    )


def test_correlated_rough_can_skip_global_noise(tmp_path: Path) -> None:
    common = {
        "model": "correlated_grf",
        "amp_range": [0.1, 0.2],
        "smoothing_sigma": [1.0, 2.0],
        "correlation_length_range": [0.10, 0.10],
        "rms_range": [0.08, 0.08],
        "fine_amplitude_fraction": 0.0,
    }
    clean_path = _write_config(
        tmp_path / "clean",
        resolution=64,
        rough={**common, "add_global_noise": False},
    )
    noisy_path = _write_config(
        tmp_path / "noisy",
        resolution=64,
        rough={**common, "add_global_noise": True},
    )
    clean = SourceGenerator(str(clean_path))
    clean.rng = np.random.default_rng(17)
    noisy = SourceGenerator(str(noisy_path))
    noisy.rng = np.random.default_rng(17)

    clean_field, _ = clean.generate()
    noisy_field, _ = noisy.generate()

    assert not np.array_equal(clean_field, noisy_field)
    assert float(np.sqrt(np.mean(clean_field * clean_field))) == pytest.approx(
        0.08, abs=1.0e-12
    )


def test_multi_gaussian_uses_exact_configured_component_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        resolution=16,
        rough={
            "model": "legacy_multiscale",
            "amp_range": [0.1, 0.2],
            "smoothing_sigma": [1.0, 2.0],
        },
        noise_enabled=False,
    )
    generator = SourceGenerator(str(path))
    generator.num_sources = np.asarray([3, 3], dtype=int)
    calls = 0

    def component() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones((16, 16), dtype=float)

    monkeypatch.setattr(generator, "_gen_gaussian_component", component)
    field = generator._gen_multi_gaussian()

    assert calls == 3
    np.testing.assert_array_equal(field, np.full((16, 16), 3.0))


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("model", "unsupported", "rough.model"),
        ("fine_amplitude_fraction", 1.1, "fine_amplitude_fraction"),
        ("min_fine_sigma_cells", 0.0, "min_fine_sigma_cells"),
        (
            "correlation_length_range",
            [0.0, 0.2],
            "correlation_length_range",
        ),
        ("rms_range", [-0.1, 0.1], "rms_range"),
    ],
)
def test_invalid_correlated_rough_config_fails_closed(
    tmp_path: Path,
    key: str,
    value: object,
    match: str,
) -> None:
    path = _write_config(
        tmp_path,
        resolution=64,
        rough={
            "model": "correlated_grf",
            "amp_range": [0.1, 0.2],
            "smoothing_sigma": [1.0, 2.0],
            "correlation_length_range": [0.10, 0.20],
            "rms_range": [0.05, 0.15],
            "fine_amplitude_fraction": 0.0,
            "add_global_noise": False,
            key: value,
        },
    )
    with pytest.raises(ValueError, match=match):
        SourceGenerator(str(path))
