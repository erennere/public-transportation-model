"""A small synthetic city for tests and demos (no downloads needed).

Residents live mostly in four suburbs, jobs concentrate in a CBD and a
secondary centre. Radial bus lines and one east-west metro exist; the
cross-town trip from the south-east suburb to the secondary centre is
deliberately badly served so the optimiser has something to find.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import Projector
from .model import City, Line, Scenario, Zones
from .traffic import ProfileTraffic

X0, Y0 = 660000.0, 4545000.0  # UTM 35N, roughly Istanbul


def _bumps(x, y, centres):
    v = np.zeros_like(x)
    for cx, cy, r, a in centres:
        v += a * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (r * r)))
    return v


def _headways(peak: float, offpeak: float) -> np.ndarray:
    h = np.full(24, np.nan)
    h[6:23] = offpeak
    h[[7, 8, 9, 17, 18, 19]] = peak
    return h


def build_synthetic(cfg: dict) -> City:
    sc = cfg.get("synthetic") or {}
    half = float(sc.get("half_size_km", 15)) * 1000
    cell = float(cfg["zones"]["cell_size_m"])
    g = np.arange(-half + cell / 2, half, cell)
    gx, gy = np.meshgrid(g, g)
    x, y = gx.ravel(), gy.ravel()
    km = 1000.0
    pop = 300 + _bumps(x, y, [(-8 * km, -6 * km, 3 * km, 9000), (9 * km, -8 * km, 3 * km, 9000),
                              (-6 * km, 9 * km, 3 * km, 7000), (8 * km, 9 * km, 2.5 * km, 6000),
                              (0, 0, 4 * km, 2000)])
    jobs = 50 + _bumps(x, y, [(0, 0, 2.5 * km, 12000), (6 * km, 4 * km, 2 * km, 7000)])
    zones = Zones(x=X0 + x, y=Y0 + y, pop=pop, jobs=jobs, cell_size=cell)

    stops_xy: list[tuple[float, float]] = []
    lines: list[Line] = []

    def add_line(lid, mode, pts, spacing, peak, off):
        pts = np.asarray(pts, float) * km
        seg = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.concatenate([[0], np.cumsum(seg)])
        d = np.linspace(0, cum[-1], max(2, int(cum[-1] / spacing) + 1))
        px, py = np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])
        idx = []
        for a, b in zip(px, py):
            idx.append(len(stops_xy))
            stops_xy.append((X0 + a, Y0 + b))
        idx = np.array(idx)
        hw = _headways(peak, off)
        lines.append(Line(lid + "_A", lid, mode, idx, hw.copy()))
        lines.append(Line(lid + "_B", lid, mode, idx[::-1].copy(), hw.copy()))

    add_line("B1", "bus", [(-8, -6), (-4, -3), (0, 0)], 400, 8, 15)
    add_line("B2", "bus", [(-6, 9), (-3, 4), (0, 0)], 400, 8, 15)
    add_line("B3", "bus", [(8, 9), (4, 4), (0, 0)], 400, 10, 20)
    add_line("B4", "bus", [(9, -8), (9, -3), (3, -1), (0, 0)], 400, 12, 20)
    add_line("M1", "metro", [(-12, 0), (0, 0), (12, 0)], 1200, 4, 8)

    sxy = np.array(stops_xy)
    stops = pd.DataFrame({"stop_id": [f"s{i}" for i in range(len(sxy))],
                          "name": [f"stop {i}" for i in range(len(sxy))],
                          "x": sxy[:, 0], "y": sxy[:, 1]})
    tc = cfg["traffic"]
    traffic = ProfileTraffic(tc["free_flow_kmh"], tc["hourly_factor"],
                             centres=[(X0, Y0, 4 * km, 0.5), (X0 + 6 * km, Y0 + 4 * km, 2.5 * km, 0.4)])
    return City(cfg=cfg, proj=Projector(32635), zones=zones, base=Scenario(stops, lines),
                traffic=traffic, hourly_profile=np.asarray(cfg["demand"]["hourly_profile"], float))
