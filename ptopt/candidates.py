"""Generate candidate new bus / metro lines where the current network fails.

1. Evaluate the base scenario at the peak hours and score each sampled trip
   by its deficit: time above Z plus how much slower transit is than car.
2. Aggregate deficits into coarse corridors (origin cell -> destination cell)
   and keep the worst ones, skipping near-duplicates.
3. For each corridor find the route through the zone grid that balances
   length against passing through dense (population + jobs) zones. This
   decides *which zones the line passes through*.
4. Buses: route through those zones on the real street network (OSM), then
   place stops along the streets at the bus stop spacing. Metro: a smoothed
   straight alignment with stations at the metro spacing.
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
    # street geometry for road-routed buses; stop_s = stop positions along it (m)
    path_xs: np.ndarray | None = None
    path_ys: np.ndarray | None = None
    stop_s: np.ndarray | None = None

    def copy(self) -> "Candidate":
        cp = lambda a: None if a is None else a.copy()
        return Candidate(self.cid, self.mode, self.xs.copy(), self.ys.copy(), self.headway_min.copy(),
                         dict(self.meta), cp(self.path_xs), cp(self.path_ys), cp(self.stop_s))

    @property
    def on_roads(self) -> bool:
        return self.path_xs is not None

    @property
    def seg_dist_m(self) -> np.ndarray | None:
        return np.diff(self.stop_s) if self.on_roads else None

    @property
    def length_km(self) -> float:
        if self.on_roads:
            return float(self.stop_s[-1] - self.stop_s[0]) / 1000.0
        return polyline_length(self.xs, self.ys) / 1000.0

    def place_stops(self) -> None:
        """Recompute stop coordinates from their positions along the street path."""
        cum = path_cumlen(self.path_xs, self.path_ys)
        self.stop_s = np.clip(np.sort(self.stop_s), 0, cum[-1])
        self.xs = np.interp(self.stop_s, cum, self.path_xs)
        self.ys = np.interp(self.stop_s, cum, self.path_ys)

    def used_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Street geometry between the first and the last stop."""
        if not self.on_roads:
            return self.xs, self.ys
        return cut_path(self.path_xs, self.path_ys, self.stop_s[0], self.stop_s[-1])


def path_cumlen(px, py) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])


def cut_path(px, py, s0: float, s1: float):
    """Part of a polyline between arc positions s0 <= s1 (interpolated ends)."""
    cum = path_cumlen(px, py)
    inner = (cum > s0) & (cum < s1)
    xs = np.concatenate([[np.interp(s0, cum, px)], px[inner], [np.interp(s1, cum, px)]])
    ys = np.concatenate([[np.interp(s0, cum, py)], py[inner], [np.interp(s1, cum, py)]])
    return xs, ys


def bus_on_roads(cid: str, roads, wx, wy, spacing: float, corridor_width: float,
                 headway: np.ndarray, meta: dict) -> "Candidate | None":
    """Bus candidate routed on streets along the corridor polyline (wx, wy)."""
    px, py = roads.route_corridor(wx, wy, width_m=corridor_width / 2)
    if len(px) < 2:
        return None
    total = path_cumlen(px, py)[-1]
    n = max(1, int(round(total / spacing)))
    c = Candidate(cid, "bus", np.zeros(0), np.zeros(0), headway, meta, px, py, np.linspace(0, total, n + 1))
    c.place_stops()
    return c


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
        seg = np.diff(c.stop_s[keep]) if c.on_roads else None
        shape = c.used_path()
        for suffix, seq, sd, shp in (("A", idx, seg, shape),
                                     ("B", idx[::-1].copy(), None if seg is None else seg[::-1].copy(),
                                      (shape[0][::-1], shape[1][::-1]))):
            meta = {"candidate": c.cid, "shape": shp}
            if sd is not None:
                meta["seg_dist_m"] = sd
            sc.lines.append(Line(f"{c.cid}_{suffix}", c.cid, c.mode, seq, c.headway_min.copy(),
                                 is_new=True, meta=meta))
    return sc


def _zone_graph(city: City, reward: float, mode: str = "metro"):
    """Zone adjacency graph whose edge cost favours dense zones.

    For buses an edge must be drivable: either every point along it lies near
    an existing stop (stops sit on roads, so this keeps lines on land), or an
    existing bus line already connects the two zones (bridges, highways)."""
    z = city.zones
    pts = np.column_stack([z.x, z.y])
    tree = cKDTree(pts)
    m = tree.sparse_distance_matrix(tree, z.cell_size * 1.5, output_type="coo_matrix")
    keep = m.row != m.col
    r, c, d = m.row[keep], m.col[keep], m.data[keep]
    if mode == "bus":
        st = city.base.stops[["x", "y"]].to_numpy()
        stree = cKDTree(st)
        ok = np.ones(len(r), bool)
        for f in (0.25, 0.5, 0.75):
            px = z.x[r] + f * (z.x[c] - z.x[r])
            py = z.y[r] + f * (z.y[c] - z.y[r])
            dist, _ = stree.query(np.column_stack([px, py]))
            ok &= dist <= 500
        r, c, d = r[ok], c[ok], d[ok]
        # zone pairs linked by consecutive stops of existing road lines
        _, zs = tree.query(st)
        er, ec = [], []
        for ln in city.base.lines:
            if ln.mode in ("bus", "brt"):
                a, b = zs[ln.stops[:-1]], zs[ln.stops[1:]]
                diff = a != b
                er.append(a[diff])
                ec.append(b[diff])
        if er:
            er, ec = np.concatenate(er), np.concatenate(ec)
            ed = np.hypot(z.x[er] - z.x[ec], z.y[er] - z.y[ec])
            r, c, d = np.concatenate([r, er, ec]), np.concatenate([c, ec, er]), np.concatenate([d, ed, ed])
    act = z.pop + z.jobs * (z.pop.sum() / max(z.jobs.sum(), 1e-12))
    dens = np.clip(act / max(np.percentile(act, 95), 1e-12), 0, 1)
    cost = d * (1.0 - reward * 0.5 * (dens[r] + dens[c]))
    G = coo_matrix((cost, (r, c)), shape=(len(z), len(z))).tocsr()
    G.sum_duplicates()
    return G


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

    graphs = {"bus": _zone_graph(city, cc["density_reward"], "bus"),
              "metro": _zone_graph(city, cc["density_reward"], "metro")}
    tree = cKDTree(np.column_stack([z.x, z.y]))
    cands: list[Candidate] = []
    for n, (ox, oy, dx, dy) in enumerate(corridors):
        oi = int(tree.query([ox, oy])[1])
        di = int(tree.query([dx, dy])[1])
        for mode, prefix, spacing, hw in (
                ("bus", "NB", cc["bus_stop_spacing_m"], cc["default_bus_headway_min"]),
                ("metro", "NM", cc["metro_station_spacing_m"], cc["default_metro_headway_min"])):
            path = _path(graphs[mode], oi, di)
            if len(path) < 2:
                continue
            ls = LineString(np.column_stack([z.x[path], z.y[path]]))
            if mode == "metro":  # straighter alignment; buses follow the drivable zone path
                ls = ls.simplify(z.cell_size * 0.75)
            px, py = np.asarray(ls.xy[0]), np.asarray(ls.xy[1])
            if mode == "metro" and polyline_length(px, py) < 5000:
                continue
            meta = {"corridor": n, "path_zones": [int(p) for p in path]}
            if mode == "bus" and city.roads is not None:
                c = bus_on_roads(f"{prefix}{n}", city.roads, px, py, spacing, cc["bus_corridor_width_m"],
                                 default_headways(hw, hw * 2), meta)
                if c is not None:
                    cands.append(c)
                continue
            sx, sy = resample_polyline(px, py, spacing)
            cands.append(Candidate(f"{prefix}{n}", mode, sx, sy, default_headways(hw, hw * 2), meta=meta))
    return cands
