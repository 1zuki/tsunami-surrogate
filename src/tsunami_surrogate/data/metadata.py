from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any


@dataclass
class ScenarioMetadata:
    sample_id: int
    source_id: int
    source_amplitude: float
    source_location: Tuple[float, float]
    bathymetry_id: int
    grid_resolution: int
    time_horizon: float
    regime: str = 'iid'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
