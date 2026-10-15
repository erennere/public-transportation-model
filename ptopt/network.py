"""Frequency-based multimodal transit graph for one hour of the day.

Nodes
    origin zones | destination zones | stops | route-stop nodes (one per line x stop)
Edges (weights in minutes)
    origin zone -> stop      walk access
    stop -> destination zone walk egress
    stop <-> stop            walking transfer
    stop -> route-stop       boarding: expected wait (headway/2, capped) + penalty
    route-stop -> stop       alighting
    route-stop -> next       in-vehicle time (traffic dependent for buses)
Origin and destination zones are separate nodes so nobody "walks through" a
zone centroid. Shortest paths are computed with scipy's sparse Dijkstra.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .model import Line, Scenario, TrafficField, Zones

EPS = 1e-6


def line_segment_minutes(line: Line, sx: np.ndarray, sy: np.ndarray, traffic: TrafficField,
                         hour: int, net: dict) -> np.ndarray:
    """In-vehicle minutes for each segment of a line at a given hour."""
    xs, ys = sx[line.stops], sy[line.stops]
    if line.seg_time_s is not None:
        return np.asarray(line.seg_time_s, float) / 60.0
    dist_m = np.hypot(np.diff(xs), np.diff(ys))
    dwell = net["dwell_s"] / 60.0
    if line.mode == "bus":
        mx, my = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
        v = np.minimum(traffic.speed_kmh(mx, my, hour) * net["bus_speed_factor"], net["bus_max_kmh"])
        road_m = line.meta.get("seg_dist_m")  # distance along real streets, when routed
        if road_m is None or len(road_m) != len(dist_m):
            road_m = dist_m * net["road_detour"]
        return np.asarray(road_m) / 1000.0 / v * 60.0 + dwell
    v = net["mode_speed_kmh"].get(line.mode, 30.0)
    return dist_m * 1.1 / 1000.0 / v * 60.0 + dwell


def line_cycle_minutes(line: Line, sx, sy, traffic, hour, net, layover: float = 0.1) -> float:
    """Round-trip time used for fleet requirements (both directions assumed)."""
    return float(2.0 * line_segment_minutes(line, sx, sy, traffic, hour, net).sum() * (1 + layover))


def line_length_km(line: Line, sx, sy) -> float:
    xs, ys = sx[line.stops], sy[line.stops]
    return float(np.hypot(np.diff(xs), np.diff(ys)).sum() / 1000.0)


class TransitGraph:
    def __init__(self, scenario: Scenario, zones: Zones, traffic: TrafficField, cfg: dict, hour: int):
        net = cfg["network"]
        self.hour = hour
        self.zones = zones
        nz = len(zones)
        sx = scenario.stops["x"].to_numpy(float)
        sy = scenario.stops["y"].to_numpy(float)
        ns = len(sx)
        walk_m_per_min = net["walk_kmh"] * 1000.0 / 60.0
        self.nz, self.ns = nz, ns
        self.zo0, self.zd0, self.s0 = 0, nz, 2 * nz
        rows, cols, w = [], [], []

        # access / egress
        stree = cKDTree(np.column_stack([sx, sy]))
        ztree = cKDTree(np.column_stack([zones.x, zones.y]))
        acc = ztree.sparse_distance_matrix(stree, net["max_access_m"], output_type="coo_matrix")
        t_acc = acc.data * net["walk_detour"] / walk_m_per_min + EPS
        rows += [self.zo0 + acc.row, self.s0 + acc.col]
        cols += [self.s0 + acc.col, self.zd0 + acc.row]
        w += [t_acc, t_acc]

        # walking transfers between stops
        pairs = stree.query_pairs(net["max_transfer_m"], output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            t = np.hypot(sx[a] - sx[b], sy[a] - sy[b]) * net["walk_detour"] / walk_m_per_min + EPS
            rows += [self.s0 + a, self.s0 + b]
            cols += [self.s0 + b, self.s0 + a]
            w += [t, t]

        # lines
        nxt = self.s0 + ns
        node_line, node_seg_km = [], []
        self.line_index = []  # lines in service, aligned with node_line ids
        for ln in scenario.lines:
            h = ln.headway_min[hour % 24]
            if not np.isfinite(h) or h <= 0 or len(ln.stops) < 2:
                continue
            li = len(self.line_index)
            self.line_index.append(ln)
            n = len(ln.stops)
            rnodes = np.arange(nxt, nxt + n)
            nxt += n
            snodes = self.s0 + ln.stops
            wait = min(h / 2.0, net["max_wait_min"]) + net["boarding_penalty_min"]
            seg = line_segment_minutes(ln, sx, sy, traffic, hour, net)
            rows += [snodes[:-1], rnodes[1:], rnodes[:-1]]
            cols += [rnodes[:-1], snodes[1:], rnodes[1:]]
            w += [np.full(n - 1, wait), np.full(n - 1, EPS), np.maximum(seg, EPS)]
            node_line.append(np.full(n, li))
            seg_km = np.hypot(np.diff(sx[ln.stops]), np.diff(sy[ln.stops])) / 1000.0
            node_seg_km.append(np.append(seg_km, 0.0))

        self.r0 = self.s0 + ns
        self.n_nodes = nxt
        self.node_line = np.concatenate(node_line) if node_line else np.zeros(0, int)
        self.node_seg_km = np.concatenate(node_seg_km) if node_seg_km else np.zeros(0)
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        ww = np.concatenate(w)
        self.G = coo_matrix((ww, (r, c)), shape=(nxt, nxt)).tocsr()

    def shortest(self, origin_zones, limit_min: float = 240.0, batch: int = 64,
                 predecessors: bool = False):
        """Travel time (min) from each origin zone to every destination zone.
        Returns times[len(origins), nz] and optionally predecessor rows."""
        origin_zones = np.asarray(origin_zones, int)
        out = np.empty((len(origin_zones), self.nz))
        preds = [] if predecessors else None
        for i in range(0, len(origin_zones), batch):
            src = self.zo0 + origin_zones[i:i + batch]
            res = dijkstra(self.G, directed=True, indices=src, limit=limit_min,
                           return_predecessors=predecessors)
            if predecessors:
                dist, pr = res
                preds.append(pr)
            else:
                dist = res
            out[i:i + batch] = dist[:, self.zd0:self.zd0 + self.nz]
        if predecessors:
            return out, np.vstack(preds)
        return out

    def line_loads(self, pred_rows: np.ndarray, origin_pos: np.ndarray, dest_zones: np.ndarray,
                   weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Trace each sampled path and accumulate (boardings, passenger-km)
        per line in service. `origin_pos[k]` indexes the predecessor row."""
        nl = len(self.line_index)
        board = np.zeros(nl)
        pkm = np.zeros(nl)
        for k in range(len(dest_zones)):
            pr = pred_rows[origin_pos[k]]
            node = self.zd0 + dest_zones[k]
            if pr[node] < 0:
                continue
            while True:
                p = pr[node]
                if p < 0:
                    break
                if node >= self.r0:
                    li = self.node_line[node - self.r0]
                    if p >= self.r0:  # ride edge p -> node
                        pkm[li] += weights[k] * self.node_seg_km[p - self.r0]
                    else:  # boarding from a stop
                        board[li] += weights[k]
                node = p
        return board, pkm
