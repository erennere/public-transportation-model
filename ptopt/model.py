"""Core data structures shared by every module."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .geo import Projector

MODES = ("bus", "brt", "tram", "metro", "rail", "ferry", "funicular", "cable")

# GTFS route_type -> internal mode
GTFS_ROUTE_TYPES = {0: "tram", 1: "metro", 2: "rail", 3: "bus", 4: "ferry", 5: "cable",
                    6: "cable", 7: "funicular", 11: "bus", 12: "metro"}


@dataclass
class Zones:
    """Analysis zones (grid cells). Coordinates are in the city's metric CRS."""
    x: np.ndarray
    y: np.ndarray
    pop: np.ndarray
    jobs: np.ndarray
    cell_size: float

    def __len__(self) -> int:
        return len(self.x)

    def to_frame(self, proj: Projector | None = None) -> pd.DataFrame:
        df = pd.DataFrame({"zone": np.arange(len(self)), "x": self.x, "y": self.y,
                           "pop": self.pop, "jobs": self.jobs})
        if proj is not None:
            df["lon"], df["lat"] = proj.to_lonlat(self.x, self.y)
        return df


@dataclass
class Line:
    """A transit line pattern (one direction).

    `stops` are positional indices into the scenario's stop table.
    `headway_min[h]` is the headway in minutes during hour h (NaN = no service).
    `seg_time_s`, when given, fixes in-vehicle time per segment (e.g. from GTFS);
    otherwise it is derived from distance and mode / traffic speed.
    """
    line_id: str
    name: str
    mode: str
    stops: np.ndarray
    headway_min: np.ndarray
    seg_time_s: np.ndarray | None = None
    is_new: bool = False
    meta: dict = field(default_factory=dict)

    def copy(self) -> "Line":
        return copy.deepcopy(self)


@dataclass
class Scenario:
    """Stops + lines. Everything an optimisation move may change."""
    stops: pd.DataFrame  # columns: stop_id, name, x, y
    lines: list[Line]

    def copy(self) -> "Scenario":
        return Scenario(self.stops.copy(), [ln.copy() for ln in self.lines])

    def add_stops(self, xs, ys, prefix: str = "new") -> np.ndarray:
        """Append stops and return their positional indices."""
        start = len(self.stops)
        n = len(xs)
        new = pd.DataFrame({"stop_id": [f"{prefix}_{start + i}" for i in range(n)],
                            "name": [f"{prefix} {start + i}" for i in range(n)],
                            "x": np.asarray(xs, float), "y": np.asarray(ys, float)})
        self.stops = pd.concat([self.stops, new], ignore_index=True)
        return np.arange(start, start + n)

    @property
    def new_lines(self) -> list[Line]:
        return [ln for ln in self.lines if ln.is_new]


class TrafficField:
    """Car speed (km/h) as a function of location and hour."""

    def speed_kmh(self, x, y, hour: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class City:
    cfg: dict
    proj: Projector
    zones: Zones
    base: Scenario
    traffic: TrafficField
    hourly_profile: np.ndarray  # share of daily trips per hour (sums to 1)
    roads: object | None = None  # ptopt.io.roads.RoadGraph, for routing new bus lines on streets
