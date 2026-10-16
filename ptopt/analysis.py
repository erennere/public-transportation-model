"""Where does door-to-door time go? Split sampled trips into walk / wait / ride."""
from __future__ import annotations

import numpy as np

from .model import City, Scenario
from .network import TransitGraph
from .simulate import UNREACHABLE_MIN, Evaluator


def time_breakdown(city: City, scenario: Scenario, hour: int = 8, K: int = 2000, n_origins: int = 200,
                   seed: int = 11) -> dict:
    """Mean minutes per component for demand-weighted trips at `hour`.

    Components: access/egress walk, transfer walk, waiting (incl. boarding
    penalty), in-vehicle, plus the share of trips that are walk-only or
    unreachable. Also returned per distance band."""
    ev = Evaluator(city, hours=[hour], K=K, M=1, n_origins=n_origins, seed=seed)
    s = ev.samples[hour][0]
    g = TransitGraph(scenario, city.zones, city.traffic, city.cfg, hour)
    origins, pos = np.unique(s.o, return_inverse=True)
    times, preds = g.shortest(origins, limit_min=UNREACHABLE_MIN, predecessors=True)
    G = g.G
    walk_only = ev.walk_t[hour][0]
    comp = {k: np.zeros(len(s.o)) for k in ("access_egress", "transfer_walk", "wait", "in_vehicle")}
    boardings = np.zeros(len(s.o))
    kind = np.empty(len(s.o), dtype=object)
    for k in range(len(s.o)):
        t_tr = times[pos[k], s.d[k]]
        if not np.isfinite(t_tr) and not np.isfinite(walk_only[k]):
            kind[k] = "unreachable"
            continue
        if walk_only[k] <= t_tr:
            kind[k] = "walk"
            comp["access_egress"][k] = walk_only[k]
            continue
        kind[k] = "transit"
        pr = preds[pos[k]]
        node = g.zd0 + s.d[k]
        while pr[node] >= 0:
            p = pr[node]
            w = G[p, node]
            if p < g.zd0 or node < g.s0:          # origin zone -> stop, stop -> destination zone
                comp["access_egress"][k] += w
            elif p < g.r0 and node < g.r0:        # stop -> stop
                comp["transfer_walk"][k] += w
            elif p < g.r0 <= node:                # boarding
                comp["wait"][k] += w
                boardings[k] += 1
            elif p >= g.r0 and node >= g.r0:      # riding
                comp["in_vehicle"][k] += w
            node = p
    z = city.zones
    dist = np.hypot(z.x[s.o] - z.x[s.d], z.y[s.o] - z.y[s.d]) / 1000
    total = np.where(kind == "unreachable", UNREACHABLE_MIN, sum(comp.values()))

    def summarise(m):
        out = {"n": int(m.sum()), "mean_total": float(total[m].mean()) if m.any() else None}
        for key, v in comp.items():
            out[key] = float(v[m].mean()) if m.any() else None
        tm = m & (kind == "transit")
        out["boardings_per_transit_trip"] = float(boardings[tm].mean()) if tm.any() else None
        out["share_walk_only"] = float((kind[m] == "walk").mean()) if m.any() else None
        out["share_unreachable"] = float((kind[m] == "unreachable").mean()) if m.any() else None
        return out

    bands = {}
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 200)):
        bands[f"{lo}-{hi} km"] = summarise((dist >= lo) & (dist < hi))
    return {"hour": hour, "all": summarise(np.ones(len(s.o), bool)), "bands": bands,
            "mean_trip_km": float(dist.mean())}
