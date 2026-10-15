"""Car speed fields by location and hour of day."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .model import TrafficField


class ProfileTraffic(TrafficField):
    """City-wide speed = free_flow * hourly_factor[h], optionally slower near
    congestion centres: factor *= 1 - depth * exp(-(d / radius)^2) at peak."""

    def __init__(self, free_flow_kmh: float, hourly_factor, centres=None):
        self.free_flow = float(free_flow_kmh)
        self.factor = np.asarray(hourly_factor, float)
        self.centres = centres or []  # [(x, y, radius_m, depth)]

    def speed_kmh(self, x, y, hour: int) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        f = self.factor[hour % 24]
        v = np.full(np.broadcast(x, y).shape, self.free_flow * f)
        congestion = 1.0 - f  # 0 off-peak, ~0.5 at peak
        for cx, cy, r, depth in self.centres:
            d2 = ((x - cx) ** 2 + (y - cy) ** 2) / (r * r)
            v = v * (1.0 - depth * congestion * np.exp(-d2))
        return np.maximum(v, 5.0)


class GridTraffic(TrafficField):
    """Observed speeds at points (e.g. geohash centres) per hour; nearest lookup.

    `speeds` has shape (n_points, 24); NaN means no observation that hour,
    which falls back to the city-wide median of that hour.
    """

    def __init__(self, x, y, speeds, max_dist_m: float = 1500.0, fallback_kmh: float = 40.0):
        self.tree = cKDTree(np.column_stack([x, y]))
        self.speeds = np.asarray(speeds, float)
        self.max_dist = max_dist_m
        med = np.nanmedian(self.speeds, axis=0)
        self.hour_median = np.where(np.isnan(med), fallback_kmh, med)

    def speed_kmh(self, x, y, hour: int) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        shape = np.broadcast(x, y).shape
        pts = np.column_stack([np.broadcast_to(x, shape).ravel(), np.broadcast_to(y, shape).ravel()])
        d, idx = self.tree.query(pts)
        h = hour % 24
        v = self.speeds[idx, h]
        v = np.where((d > self.max_dist) | np.isnan(v), self.hour_median[h], v)
        return np.maximum(v, 5.0).reshape(shape)


def ibb_hourly_speeds(csv_paths, proj, weekdays_only: bool = True, chunksize: int = 2_000_000):
    """Aggregate İBB 'hourly traffic density' CSVs (one per month) into
    mean speed per geohash and hour of day. Returns (x, y, speeds[n, 24])."""
    usecols = ["DATE_TIME", "LATITUDE", "LONGITUDE", "GEOHASH", "AVERAGE_SPEED", "NUMBER_OF_VEHICLES"]
    parts = []
    for path in csv_paths:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, encoding="utf-8-sig"):
            t = pd.to_datetime(chunk["DATE_TIME"], errors="coerce")
            if weekdays_only:
                keep = t.dt.dayofweek < 5
                chunk, t = chunk[keep], t[keep]
            chunk = chunk.assign(hour=t.dt.hour,
                                 w=chunk["NUMBER_OF_VEHICLES"].clip(lower=1),
                                 sw=chunk["AVERAGE_SPEED"] * chunk["NUMBER_OF_VEHICLES"].clip(lower=1))
            g = chunk.groupby(["GEOHASH", "hour"]).agg(sw=("sw", "sum"), w=("w", "sum"),
                                                       lat=("LATITUDE", "first"),
                                                       lon=("LONGITUDE", "first"))
            parts.append(g)
    allg = pd.concat(parts).groupby(level=[0, 1]).agg(sw=("sw", "sum"), w=("w", "sum"),
                                                      lat=("lat", "first"), lon=("lon", "first"))
    allg["v"] = allg["sw"] / allg["w"]
    table = allg["v"].unstack("hour").reindex(columns=range(24))
    loc = allg[["lat", "lon"]].groupby(level=0).first().loc[table.index]
    x, y = proj.to_xy(loc["lon"].to_numpy(), loc["lat"].to_numpy())
    return x, y, table.to_numpy()


def car_time_min(traffic: TrafficField, x1, y1, x2, y2, hour: int, cfg_net: dict, n_samples: int = 7):
    """Approximate door-to-door car time: straight line * detour at the harmonic
    mean of local speeds along the way, plus access/parking time."""
    x1, y1, x2, y2 = (np.asarray(a, float) for a in (x1, y1, x2, y2))
    s = np.linspace(0.05, 0.95, n_samples)[:, None]
    xs = x1[None, :] + s * (x2 - x1)[None, :]
    ys = y1[None, :] + s * (y2 - y1)[None, :]
    v = traffic.speed_kmh(xs, ys, hour)
    v_h = n_samples / np.sum(1.0 / v, axis=0)
    dist_km = np.hypot(x2 - x1, y2 - y1) * cfg_net["road_detour"] / 1000.0
    return dist_km / v_h * 60.0 + cfg_net["car_access_min"]
