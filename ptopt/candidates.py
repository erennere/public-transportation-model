"""Generate candidate new bus / metro lines where the current network fails.

1. Evaluate the base scenario at the peak hours and score each sampled trip
   by its deficit: time above Z plus how much slower transit is than car.
2. Aggregate deficits into coarse corridors (origin cell -> destination cell)
   and keep the worst ones, skipping near-duplicates.
3. For each corridor find the route through the zone grid that balances
   length against passing through dense (population + jobs) zones. This
   decides *which zones the line passes through*.
4. Place stops along that path at the mode's spacing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from .geo import polyline_length, resample_polyline
from .model import City, Line, Scenario
from .simulate import Evaluator


@dataclass
class Candidate:
    cid: str
    mode: str          # 'bus' | 'metro'
    xs: np.ndarray     # stop positions (metric CRS)
    ys: np.ndarray
    headway_min: np.ndarray  # 24 values, NaN = no service
    meta: dict = field(default_factory=dict)

    def copy(self) -> "Candidate":
        return Candidate(self.cid, self.mode, self.xs.copy(), self.ys.copy(),
                         self.headway_min.copy(), dict(self.meta))

    @property
    def length_km(self) -> float:
        return polyline_length(self.xs, self.ys) / 1000.0


def default_headways(peak: float, off: float, peak_hours=(7, 8, 9, 17, 18, 19)) -> np.ndarray:
    h = np.full(24, np.nan)
    h[6:24] = off
    h[list(peak_hours)] = peak
    return h


def materialize(base: Scenario, cands: list[Candidate], snap_m: float) -> Scenario:
    """Base scenario plus the given candidates as two-directional lines."""
    sc = base.copy()
    n_base = len(base.stops)
    tree = cKDTree(base.stops[["x", "y"]].to_numpy()) if n_base else None
    for c in cands:
        idx = np.empty(len(c.xs), dtype=int)
        new_mask = np.ones(len(c.xs), bool)
        if c.mode == "bus" and tree is not None:
            d, j = tree.query(np.column_stack([c.xs, c.ys]))
            snap = d <= snap_m
            idx[snap] = j[snap]
            new_mask = ~snap
        if new_mask.any():
            idx[new_mask] = sc.add_stops(c.xs[new_mask], c.ys[new_mask], prefix=c.cid)
        # consecutive duplicates (two points snapped to the same stop)
        keep = np.concatenate([[True], idx[1:] != idx[:-1]])
        idx = idx[keep]
        if len(idx) < 2:
            continue
        for suffix, seq in (("A", idx), ("B", idx[::-1].copy())):
            sc.lines.append(Line(f"{c.cid}_{suffix}", c.cid, c.mode, seq, c.headway_min.copy(),
                                 is_new=True, meta={"candidate": c.cid}))
    return sc


def _zone_graph(city: City, reward: float):
    z = city.zones
    pts = np.column_stack([z.x, z.y])
    tree = cKDTree(pts)
    m = tree.sparse_distance_matrix(tree, z.cell_size * 1.5, output_type="coo_matrix")
    keep = m.row != m.col
    r, c, d = m.row[keep], m.col[keep], m.data[keep]
    act = z.pop + z.jobs * (z.pop.sum() / max(z.jobs.sum(), 1e-12))
    dens = np.clip(act / max(np.percentile(act, 95), 1e-12), 0, 1)
    cost = d * (1.0 - reward * 0.5 * (dens[r] + dens[c]))
    return coo_matrix((cost, (r, c)), shape=(len(z), len(z))).tocsr()


def _path(G, o: int, d: int) -> list[int]:
    _, pred = dijkstra(G, directed=False, indices=o, return_predecessors=True)
    if pred[d] < 0 and o != d:
        return []
    path = [d]
    while path[-1] != o:
        path.append(pred[path[-1]])
    return path[::-1]


def corridor_deficits(city: City, ev: Evaluator, scenario: Scenario | None = None):
    """Per sampled trip deficit at the evaluator's hours. Returns arrays
    (o, d, deficit)."""
    scenario = scenario or city.base
    res = ev.evaluate(scenario, keep_times=True)
    os_, ds, defs = [], [], []
    i = 0
    for h in ev.hours:
        for m, s in enumerate(ev.samples[h]):
            t = res.sims[i].t_pt
            i += 1
            car = ev.car_t[h][m]
            deficit = s.weight * (np.maximum(0, t - ev.Z) + 0.5 * np.maximum(0, t - car))
            os_.append(s.o)
            ds.append(s.d)
            defs.append(deficit)
    return np.concatenate(os_), np.concatenate(ds), np.concatenate(defs)


def generate_candidates(city: City, ev: Evaluator, scenario: Scenario | None = None) -> list[Candidate]:
    cc = city.cfg["candidates"]
    z = city.zones
    o, d, deficit = corridor_deficits(city, ev, scenario)
    coarse = 3000.0
    cx = np.floor((z.x - z.x.min()) / coarse).astype(int)
    cy = np.floor((z.y - z.y.min()) / coarse).astype(int)
    cell = cx * 100000 + cy
    key = cell[o].astype(np.int64) * 10**10 + cell[d]
    order = np.argsort(key)
    key_s, o_s, d_s, def_s = key[order], o[order], d[order], deficit[order]
    uniq, start = np.unique(key_s, return_index=True)
    sums = np.add.reduceat(def_s, start)
    corridors = []
    for k in np.argsort(-sums):
        if sums[k] <= 0:
            break
        sl = slice(start[k], start[k + 1] if k + 1 < len(start) else len(key_s))
        w = def_s[sl]
        # representative endpoints: deficit-weighted centroid -> nearest zone
        ox = np.average(z.x[o_s[sl]], weights=w)
        oy = np.average(z.y[o_s[sl]], weights=w)
        dx = np.average(z.x[d_s[sl]], weights=w)
        dy = np.average(z.y[d_s[sl]], weights=w)
        if np.hypot(dx - ox, dy - oy) < cc["min_od_km"] * 1000:
            continue
        dup = any(min(np.hypot(ox - a, oy - b) + np.hypot(dx - c_, dy - e),
                      np.hypot(ox - c_, oy - e) + np.hypot(dx - a, dy - b)) < 5000
                  for a, b, c_, e in corridors)
        if dup:
            continue
        corridors.append((ox, oy, dx, dy))
        if len(corridors) >= cc["n_corridors"]:
            break

    G = _zone_graph(city, cc["density_reward"])
    tree = cKDTree(np.column_stack([z.x, z.y]))
    cands: list[Candidate] = []
    for n, (ox, oy, dx, dy) in enumerate(corridors):
        oi = int(tree.query([ox, oy])[1])
        di = int(tree.query([dx, dy])[1])
        path = _path(G, oi, di)
        if len(path) < 2:
            continue
        ls = LineString(np.column_stack([z.x[path], z.y[path]])).simplify(z.cell_size * 0.75)
        px, py = np.asarray(ls.xy[0]), np.asarray(ls.xy[1])
        length = polyline_length(px, py)
        bx, by = resample_polyline(px, py, cc["bus_stop_spacing_m"])
        cands.append(Candidate(f"NB{n}", "bus", bx, by,
                               default_headways(cc["default_bus_headway_min"],
                                                cc["default_bus_headway_min"] * 2),
                               meta={"corridor": n, "path_zones": [int(p) for p in path]}))
        if length >= 5000:
            mx, my = resample_polyline(px, py, cc["metro_station_spacing_m"])
            cands.append(Candidate(f"NM{n}", "metro", mx, my,
                                   default_headways(cc["default_metro_headway_min"],
                                                    cc["default_metro_headway_min"] * 2),
                                   meta={"corridor": n, "path_zones": [int(p) for p in path]}))
    return cands
