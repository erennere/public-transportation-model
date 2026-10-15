"""Time-of-day origin-destination demand from population and jobs.

Residents (WorldPop) generate trips, jobs (OSM POI proxy) attract them.
Home->work and work->home matrices come from a doubly constrained gravity
model; the hour of day decides how much of each purpose is travelling, so
in the morning flows run from residential zones X to job zones Y and in the
evening the other way round.
"""
from __future__ import annotations

import numpy as np

from .model import Zones


def gravity(origins, destinations, dist_km, d0_km: float, iterations: int = 30) -> np.ndarray:
    """Doubly constrained gravity model (Furness balancing), diagonal excluded.
    Returns T with row sums ~ origins and column sums ~ destinations (both
    scaled to the same total)."""
    o = np.asarray(origins, float)
    d = np.asarray(destinations, float)
    d = d * (o.sum() / max(d.sum(), 1e-12))
    f = np.exp(-dist_km / d0_km)
    np.fill_diagonal(f, 0.0)
    a = np.ones_like(o)
    b = np.ones_like(d)
    for _ in range(iterations):
        a = 1.0 / np.maximum(f @ (b * d), 1e-12)
        b = 1.0 / np.maximum((a * o) @ f, 1e-12)
    return (a * o)[:, None] * f * (b * d)[None, :]


class Demand:
    """Hourly OD matrices T(h) in trips/hour between zones."""

    def __init__(self, zones: Zones, cfg: dict, hourly_profile: np.ndarray):
        dc = cfg["demand"]
        self.zones = zones
        dx = zones.x[:, None] - zones.x[None, :]
        dy = zones.y[:, None] - zones.y[None, :]
        self.dist_km = np.hypot(dx, dy) / 1000.0
        hw = gravity(zones.pop, zones.jobs, self.dist_km, dc["gravity_d0_km"], dc["furness_iterations"])
        act = zones.pop + zones.jobs * (zones.pop.sum() / max(zones.jobs.sum(), 1e-12))
        other = gravity(act, act, self.dist_km, dc["gravity_d0_km"] * 0.6, dc["furness_iterations"])
        self.m_hw = hw / hw.sum()
        self.m_other = other / other.sum()
        self.daily_trips = zones.pop.sum() * dc["daily_trips_per_person"] * dc["transit_capture"]
        self.profile = np.asarray(hourly_profile, float) / np.sum(hourly_profile)
        self.s_hw = np.asarray(dc["share_home_work"], float)
        self.s_wh = np.asarray(dc["share_work_home"], float)

    def purpose_shares(self, hour: int) -> tuple[float, float, float]:
        h = hour % 24
        hw, wh = self.s_hw[h], self.s_wh[h]
        return hw, wh, max(0.0, 1.0 - hw - wh)

    def matrix(self, hour: int) -> np.ndarray:
        hw, wh, ot = self.purpose_shares(hour)
        m = hw * self.m_hw + wh * self.m_hw.T + ot * self.m_other
        return self.daily_trips * self.profile[hour % 24] * m
