"""Parameters shared by CLI, desktop demo, and evaluation."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import math


@dataclass(frozen=True)
class Config:
    sigma: float = 1.0
    gaussian_tail: float = 0.01
    low_threshold: float = 30.0
    high_threshold: float = 70.0
    closing_iterations: int = 1
    min_area: int = 200
    max_area_fraction: float = 0.45
    min_hole_fraction: float = 0.015
    max_hole_fraction: float = 0.50
    hole_background_tolerance: float = 0.50
    bolt_min_elongation: float = 2.2
    round_max_elongation: float = 1.8
    washer_max_radial_cv: float = 0.035
    nut_min_hex_score: float = 0.025
    nut_max_radial_cv: float = 0.15

    def __post_init__(self):
        if any(not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in asdict(self).values()):
            raise ValueError("All parameters must be finite numbers")
        if self.sigma < 0.5 or not 0 < self.gaussian_tail < 1:
            raise ValueError("sigma >= 0.5 and 0 < gaussian_tail < 1 required")
        if not 0 < self.low_threshold <= self.high_threshold <= 255:
            raise ValueError("Require 0 < low_threshold <= high_threshold <= 255")
        if (not isinstance(self.closing_iterations, int) or self.closing_iterations < 0
                or not isinstance(self.min_area, int) or self.min_area < 1):
            raise ValueError("closing_iterations >= 0 and min_area >= 1 must be integers")
        if not 0 < self.max_area_fraction <= 1:
            raise ValueError("max_area_fraction must be in (0, 1]")
        if not 0 < self.min_hole_fraction < self.max_hole_fraction < 1:
            raise ValueError("Invalid hole area fractions")
        if not 0 < self.hole_background_tolerance < 1:
            raise ValueError("hole_background_tolerance must be in (0, 1)")
        if not 1 < self.round_max_elongation < self.bolt_min_elongation:
            raise ValueError("Require 1 < round_max_elongation < bolt_min_elongation")
        if not (0 < self.washer_max_radial_cv < self.nut_max_radial_cv
                and 0 < self.nut_min_hex_score < 1):
            raise ValueError("Invalid radial shape thresholds")

    def to_dict(self):
        return asdict(self)


def load_config(path=None):
    if path is None:
        return Config()
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    allowed = {f.name for f in fields(Config)}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Config must be an object containing known parameter names")
    return Config(**data)
