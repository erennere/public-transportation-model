"""Calibrate the model against observed travel times (Google Routes, TomTom).

Car: for a sample of trips we know the model's car time and the providers'
traffic-aware time. Each trip gives a speed multiplier (how much faster or
slower traffic really is along it); multipliers are smoothed in space (k nearest
trips) per hour and applied on top of the İBB speed field. Buses inherit the
corrected speeds.

Transit: model parameters (bus speed factor, boarding penalty, wait cap, walk
detour, access radius, rail speed) are fitted with Nelder-Mead so the model's
door-to-door transit times match Google's transit times in log terms.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .demand import Demand
from .io.external import Cache, google_matrix, tomtom_drive
from .model import City, TrafficField
from .network import TransitGraph
from .sampling import sample_od
from .traffic import car_time_min

log = logging.getLogger(__name__)

# name, config key path, lower, upper
TRANSIT_PARAMS = [
    ("bus_speed_factor", ("network", "bus_speed_factor"), 0.45, 1.3),
    ("boarding_penalty_min", ("network", "boarding_penalty_min"), 0.0, 6.0),
    ("max_wait_min", ("network", "max_wait_min"), 4.0, 20.0),
    ("walk_detour", ("network", "walk_detour"), 1.1, 1.6),
    ("max_access_m", ("network", "max_access_m"), 500.0, 1500.0),
    ("rail_speed_scale", ("network", "rail_speed_scale"), 0.7, 1.4),
]


class CalibratedTraffic(TrafficField):
    """Base speed field x locally smoothed multipliers fitted to observed car times."""

    def __init__(self, base: TrafficField, points: dict, global_factor: float, k: int = 10):
        self.base = base
        self.global_factor = float(global_factor)
        self.k = k
        self.hours = {int(h): (cKDTree(np.column_stack([p["x"], p["y"]])), np.asarray(p["f"], float))
                      for h, p in points.items() if len(p["x"]) >= 3}

    def factor(self, x, y, hour: int) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        shape = np.broadcast(x, y).shape
        if not self.hours:
            return np.full(shape, self.global_factor)
        h = min(self.hours, key=lambda c: abs(c - hour % 24))
        tree, f = self.hours[h]
        pts = np.column_stack([np.broadcast_to(x, shape).ravel(), np.broadcast_to(y, shape).ravel()])
        k = min(self.k, len(f))
        d, i = tree.query(pts, k=k)
        w = 1.0 / np.maximum(d, 500.0) ** 2
        val = np.exp((w * np.log(f[i])).sum(1) / w.sum(1))
        if h != hour % 24:  # outside the calibrated hours only the overall level is transferred
            val = np.full_like(val, self.global_factor)
        return val.reshape(shape)

    def speed_kmh(self, x, y, hour: int) -> np.ndarray:
        return np.maximum(self.base.speed_kmh(x, y, hour) * self.factor(x, y, hour), 3.0)


@dataclass
class Samples:
    hour: np.ndarray
    o: np.ndarray
    d: np.ndarray
    o_ll: np.ndarray  # (n, 2) lon/lat
    d_ll: np.ndarray
    car_obs: np.ndarray      # minutes, NaN if unknown
    transit_obs: np.ndarray


def sample_pairs(city: City, n_per_hour: int, hours, seed: int = 7, min_km: float = 1.5):
    """Demand-weighted OD sample; points are the street node nearest to each zone centre."""
    dem = Demand(city.zones, city.cfg, city.hourly_profile)
    z = city.zones
    rows = []
    for h in hours:
        T = dem.matrix(h)
        s = sample_od(T, n_per_hour * 3, n_per_hour, np.random.default_rng([seed, h]))
        dist = np.hypot(z.x[s.o] - z.x[s.d], z.y[s.o] - z.y[s.d])
        keep = np.flatnonzero(dist >= min_km * 1000)
        seen, picked = set(), []
        for k in keep:  # distinct pairs
            if (s.o[k], s.d[k]) not in seen:
                seen.add((s.o[k], s.d[k]))
                picked.append(k)
            if len(picked) == n_per_hour:
                break
        rows += [(h, s.o[k], s.d[k]) for k in picked]
    hour = np.array([r[0] for r in rows])
    o = np.array([r[1] for r in rows])
    d = np.array([r[2] for r in rows])

    def snap(idx):
        x, y = z.x[idx], z.y[idx]
        if city.roads is not None:
            n = city.roads.nearest(x, y)
            x, y = city.roads.x[n], city.roads.y[n]
        lon, lat = city.proj.to_lonlat(x, y)
        return np.column_stack([lon, lat])

    nan = np.full(len(rows), np.nan)
    return Samples(hour, o, d, snap(o), snap(d), nan.copy(), nan.copy())


def collect(city: City, s: Samples, cache_path: Path, max_google: int = 4000, max_tomtom: int = 2000):
    """Query the providers (or the cache) and fill the observed times."""
    cache = Cache(cache_path)
    gdrive = np.full(len(s.o), np.nan)
    tdrive = np.full(len(s.o), np.nan)
    gtransit = np.full(len(s.o), np.nan)
    for h in np.unique(s.hour):
        idx = np.flatnonzero(s.hour == h)
        pairs = [(tuple(s.o_ll[i]), tuple(s.d_ll[i])) for i in idx]
        for arr, vals in ((gdrive, google_matrix(pairs, "DRIVE", int(h), cache, max_elements=max_google // 2)),
                          (gtransit, google_matrix(pairs, "TRANSIT", int(h), cache, max_elements=max_google // 2)),
                          (tdrive, tomtom_drive(pairs, int(h), cache, max_requests=max_tomtom))):
            arr[idx] = [np.nan if v is None else v for v in vals]
    s.car_obs = np.nanmedian(np.vstack([gdrive, tdrive]), axis=0)
    s.transit_obs = gtransit
    return {"google_drive": gdrive, "tomtom_drive": tdrive, "google_transit": gtransit}


def _err_stats(model, obs) -> dict:
    ok = np.isfinite(model) & np.isfinite(obs) & (obs > 0)
    if not ok.any():
        return {"n": 0}
    r = model[ok] / obs[ok]
    return {"n": int(ok.sum()), "median_ratio": float(np.median(r)),
            "median_abs_pct_err": float(np.median(np.abs(r - 1)) * 100),
            "mean_model_min": float(model[ok].mean()), "mean_obs_min": float(obs[ok].mean())}


def fit_car(city: City, s: Samples) -> tuple[CalibratedTraffic, dict]:
    net = city.cfg["network"]
    z = city.zones
    ox, oy, dx, dy = z.x[s.o], z.y[s.o], z.x[s.d], z.y[s.d]
    before = np.array([car_time_min(city.traffic, [ox[i]], [oy[i]], [dx[i]], [dy[i]], int(s.hour[i]), net)[0]
                       for i in range(len(s.o))])
    acc = net["car_access_min"]
    ok = np.isfinite(s.car_obs) & (s.car_obs > acc + 1)
    # multiplier on speed so that model drive time matches the observation
    f = np.clip((before - acc) / np.maximum(s.car_obs - acc, 1e-6), 0.3, 3.0)
    points = {}
    for h in np.unique(s.hour[ok]):
        m = ok & (s.hour == h)
        points[int(h)] = {"x": ((ox + dx) / 2)[m].tolist(), "y": ((oy + dy) / 2)[m].tolist(), "f": f[m].tolist()}
    g = float(np.exp(np.median(np.log(f[ok])))) if ok.any() else 1.0
    traffic = CalibratedTraffic(city.traffic, points, g)
    after = np.array([car_time_min(traffic, [ox[i]], [oy[i]], [dx[i]], [dy[i]], int(s.hour[i]), net)[0]
                      for i in range(len(s.o))])
    stats = {"before": _err_stats(before, s.car_obs), "after": _err_stats(after, s.car_obs),
             "global_speed_factor": g}
    return traffic, {"points": points, "global_factor": g, "stats": stats}


def _set(cfg, path, value):
    d = cfg
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def transit_times(city: City, s: Samples, cfg: dict, traffic: TrafficField) -> np.ndarray:
    z = city.zones
    net = cfg["network"]
    walk_m_per_min = net["walk_kmh"] * 1000.0 / 60.0
    out = np.full(len(s.o), np.nan)
    for h in np.unique(s.hour):
        idx = np.flatnonzero(s.hour == h)
        g = TransitGraph(city.base, z, traffic, cfg, int(h))
        origins, pos = np.unique(s.o[idx], return_inverse=True)
        t = g.shortest(origins, limit_min=240)[pos, s.d[idx]]
        dist = np.hypot(z.x[s.o[idx]] - z.x[s.d[idx]], z.y[s.o[idx]] - z.y[s.d[idx]])
        walk = np.where(dist <= net["max_walk_only_m"], dist * net["walk_detour"] / walk_m_per_min, np.inf)
        out[idx] = np.minimum(t, walk)
    return np.where(np.isfinite(out), out, 240.0)


def fit_transit(city: City, s: Samples, traffic: TrafficField, max_evals: int = 50) -> dict:
    ok = np.isfinite(s.transit_obs) & (s.transit_obs > 0)
    base_cfg = copy.deepcopy(city.cfg)
    base_cfg["network"].setdefault("rail_speed_scale", 1.0)
    lo = np.array([p[2] for p in TRANSIT_PARAMS])
    hi = np.array([p[3] for p in TRANSIT_PARAMS])
    x0 = np.array([float(base_cfg[p[1][0]][p[1][1]]) for p in TRANSIT_PARAMS])
    x0 = np.clip(x0, lo, hi)

    def to_cfg(u):
        v = lo + (hi - lo) / (1 + np.exp(-u))  # unconstrained -> bounded
        cfg = copy.deepcopy(base_cfg)
        for p, val in zip(TRANSIT_PARAMS, v):
            _set(cfg, p[1], float(val))
        return cfg, v

    def u_of(v):
        q = np.clip((v - lo) / (hi - lo), 1e-3, 1 - 1e-3)
        return np.log(q / (1 - q))

    trace = []

    def loss(u):
        cfg, v = to_cfg(u)
        t = transit_times(city, s, cfg, traffic)
        e = np.log(t[ok] / s.transit_obs[ok])
        val = float(np.mean(np.minimum(e ** 2, 1.0)))  # robust to a few unmatched trips
        trace.append((val, v.tolist()))
        log.info("transit fit %d: loss %.4f  %s", len(trace), val,
                 {p[0]: round(x, 2) for p, x in zip(TRANSIT_PARAMS, v)})
        return val

    t_before = transit_times(city, s, base_cfg, traffic)
    res = minimize(loss, u_of(x0), method="Nelder-Mead",
                   options={"maxfev": max_evals, "xatol": 0.05, "fatol": 1e-4, "initial_simplex": None})
    cfg_best, v_best = to_cfg(res.x)
    t_after = transit_times(city, s, cfg_best, traffic)
    return {"params": {p[0]: float(x) for p, x in zip(TRANSIT_PARAMS, v_best)},
            "stats": {"before": _err_stats(t_before, s.transit_obs), "after": _err_stats(t_after, s.transit_obs)},
            "evals": len(trace)}


def apply_calibration(city: City, calib: dict) -> City:
    """Use fitted parameters and the corrected speed field (in place)."""
    for name, value in (calib.get("transit") or {}).get("params", {}).items():
        path = next(p[1] for p in TRANSIT_PARAMS if p[0] == name)
        _set(city.cfg, path, value)
    car = calib.get("car")
    if car:
        city.traffic = CalibratedTraffic(city.traffic, car["points"], car["global_factor"])
    return city


def save(calib: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=2, default=float))


def load(path: Path) -> dict | None:
    return json.loads(Path(path).read_text()) if Path(path).exists() else None
