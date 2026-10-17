"""Budget–time frontier: how far can a given investment push travel times down?

Investment options ("actions"):
  line    a new bus line (on streets) or a new metro line
  extend  extension of an existing metro / rail line at one end (through service,
          no extra transfer)
  boost   more buses on a busy existing bus route (shorter headway)

Budgets are relative to what exists: extra buses as a share of today's peak bus
fleet, new rail km as a share of today's rail network length.

A greedy search adds, step by step, the action with the best objective gain per
unit of (normalised) cost. Every candidate is first screened with a cheap
evaluation; the best few are re-evaluated with a larger sample before one is
picked. Candidates are regenerated from the current network as it grows, so new
gaps are found once old ones are filled. The recorded steps form the frontier.
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .candidates import Candidate, generate_candidates, materialize
from .geo import resample_polyline
from .model import City, Line, Scenario
from .network import line_cycle_minutes, line_length_km
from .simulate import Evaluator

log = logging.getLogger(__name__)

RAIL_MODES = ("metro", "rail", "tram", "funicular")
ROAD_MODES = ("bus", "brt")
_WORKER = None


@dataclass
class Action:
    aid: str
    kind: str  # 'line' | 'extend' | 'boost'
    cost_buses: int
    cost_km: float
    cand: Candidate | None = None
    route: str | None = None
    factor: float = 1.0
    meta: dict = field(default_factory=dict)

    def label(self) -> str:
        if self.kind == "line":
            return f"yeni {'metro' if self.cand.mode == 'metro' else 'otobüs'} {self.aid} ({self.cand.length_km:.1f} km)"
        if self.kind == "extend":
            return f"{_short(self.route)} uzatma {self.aid} (+{self.cost_km:.1f} km)"
        return f"{_short(self.route)} sıklık ×{1 / self.factor:.2f} (+{self.cost_buses} otobüs)"


def _short(name: str) -> str:
    return name.split(":", 1)[-1]


# ---------------------------------------------------------------- inventory
def inventory(city: City, scenario: Scenario | None = None, hour: int = 8) -> dict:
    """Peak bus fleet and rail network length of a scenario."""
    sc = scenario or city.base
    net = city.cfg["network"]
    sx, sy = sc.stops["x"].to_numpy(), sc.stops["y"].to_numpy()
    fleet = 0.0
    for ln in sc.lines:
        h = ln.headway_min[hour]
        if ln.mode in ROAD_MODES and np.isfinite(h) and h > 0:
            fleet += line_cycle_minutes(ln, sx, sy, city.traffic, hour, net) / 2 / h  # one direction
    rail: dict[str, float] = {}
    for ln in sc.lines:
        if ln.mode in RAIL_MODES:
            rail[ln.name] = max(rail.get(ln.name, 0.0), line_length_km(ln, sx, sy))
    return {"bus_fleet_peak": float(fleet), "rail_km": float(sum(rail.values())), "rail_lines": len(rail)}


# ---------------------------------------------------------------- applying actions
def apply_actions(city: City, actions: list[Action], base: Scenario | None = None) -> Scenario:
    snap = city.cfg["candidates"]["snap_existing_m"]
    sc = materialize(base or city.base, [a.cand for a in actions if a.kind == "line"], snap)
    for a in actions:
        if a.kind == "extend":
            _extend(sc, a)
        elif a.kind == "boost":
            for ln in sc.lines:
                if ln.name == a.route:
                    ln.headway_min = ln.headway_min * a.factor
    return sc


def _extend(sc: Scenario, a: Action) -> None:
    tx, ty = a.meta["terminal"]
    sx, sy = sc.stops["x"].to_numpy(), sc.stops["y"].to_numpy()
    new_idx = None
    for ln in sc.lines:
        if ln.name != a.route:
            continue
        d_end = math.hypot(sx[ln.stops[-1]] - tx, sy[ln.stops[-1]] - ty)
        d_start = math.hypot(sx[ln.stops[0]] - tx, sy[ln.stops[0]] - ty)
        if min(d_end, d_start) > 300:
            continue
        if new_idx is None:
            new_idx = sc.add_stops(a.cand.xs, a.cand.ys, prefix=a.aid)
        if d_end <= d_start:
            ln.stops = np.concatenate([ln.stops, new_idx])
        else:
            ln.stops = np.concatenate([new_idx[::-1], ln.stops])
        ln.seg_time_s = None
        ln.meta = {**ln.meta, "extensions": ln.meta.get("extensions", []) + [a.aid]}
        ln.meta.setdefault("ext_stops", [])
        ln.meta["ext_stops"] = ln.meta["ext_stops"] + [int(i) for i in new_idx]


# ---------------------------------------------------------------- generating actions
def _bus_cost(city: City, lines: list[Line], factor: float, hour: int, sc: Scenario) -> int:
    net = city.cfg["network"]
    sx, sy = sc.stops["x"].to_numpy(), sc.stops["y"].to_numpy()
    extra = 0.0
    for ln in lines:
        h = ln.headway_min[hour]
        if np.isfinite(h) and h > 0:
            c = line_cycle_minutes(ln, sx, sy, city.traffic, hour, net) / 2
            extra += c / (h * factor) - c / h
    return int(math.ceil(extra))


def line_actions(city: City, cands: list[Candidate], hour: int) -> list[Action]:
    out = []
    net = city.cfg["network"]
    for c in cands:
        if c.mode == "metro":
            out.append(Action(c.cid, "line", 0, c.length_km, cand=c))
        else:
            ln = Line(c.cid, c.cid, c.mode, np.arange(len(c.xs)), c.headway_min,
                      meta={} if c.seg_dist_m is None else {"seg_dist_m": c.seg_dist_m})
            cyc = line_cycle_minutes(ln, c.xs, c.ys, city.traffic, hour, net)
            out.append(Action(c.cid, "line", int(math.ceil(cyc / np.nanmin(c.headway_min))), 0.0, cand=c))
    return out


def extension_actions(city: City, sc: Scenario, step: int, lengths=(4000, 8000),
                      cone_deg: float = 60.0) -> list[Action]:
    """For each end of each metro/rail line: extend 4 or 8 km towards the densest
    zones ahead (activity within 1.5 km of the extension, per km)."""
    z = city.zones
    act = z.pop + z.jobs * (z.pop.sum() / max(z.jobs.sum(), 1e-12))
    ztree = cKDTree(np.column_stack([z.x, z.y]))
    sx, sy = sc.stops["x"].to_numpy(), sc.stops["y"].to_numpy()
    spacing = city.cfg["candidates"]["metro_station_spacing_m"]
    longest: dict[str, Line] = {}
    for ln in sc.lines:
        if ln.mode in ("metro", "rail") and (ln.name not in longest or len(ln.stops) > len(longest[ln.name].stops)):
            longest[ln.name] = ln
    out = []
    for name, ln in longest.items():
        if len(ln.stops) < 3:
            continue
        for end in (0, -1):
            t = ln.stops[end]
            p = ln.stops[min(2, len(ln.stops) - 1)] if end == 0 else ln.stops[max(-3, -len(ln.stops))]
            ux, uy = sx[t] - sx[p], sy[t] - sy[p]
            n = math.hypot(ux, uy) or 1.0
            ux, uy = ux / n, uy / n
            best = None
            for ang in np.radians(np.linspace(-cone_deg, cone_deg, 9)):
                dx = ux * math.cos(ang) - uy * math.sin(ang)
                dy = ux * math.sin(ang) + uy * math.cos(ang)
                for L in lengths:
                    xs, ys = resample_polyline([sx[t], sx[t] + dx * L], [sy[t], sy[t] + dy * L], spacing)
                    near = ztree.query_ball_point(np.column_stack([xs, ys]), 1500)
                    zs = set(i for lst in near for i in lst)
                    score = act[list(zs)].sum() / (L / 1000) if zs else 0.0
                    if best is None or score > best[0]:
                        best = (score, xs[1:], ys[1:], L)
            if best is None or best[0] <= 0:
                continue
            _, xs, ys, L = best
            aid = f"X{step}_{_short(name)}_{'a' if end == 0 else 'b'}"
            cand = Candidate(aid, "metro", np.asarray(xs), np.asarray(ys), ln.headway_min.copy())
            out.append(Action(aid, "extend", 0, L / 1000, cand=cand, route=name,
                              meta={"terminal": (float(sx[t]), float(sy[t]))}))
    return out


def boost_actions(city: City, sc: Scenario, ev: Evaluator, hour: int, top: int, factor: float,
                  min_headway: float) -> list[Action]:
    """More buses on the busiest bus routes (by boardings at the peak hour)."""
    res = ev.evaluate(sc, loads=True)
    board: dict[str, float] = {}
    by_id = {ln.line_id: ln for ln in sc.lines}
    for s in res.sims:
        for lid, (b, _) in (s.loads or {}).items():
            ln = by_id.get(lid)
            if ln is not None and ln.mode in ROAD_MODES:
                board[ln.name] = board.get(ln.name, 0.0) + b / len(res.sims)
    out = []
    for name in sorted(board, key=board.get, reverse=True):
        if len(out) >= top:  # the `top` busiest routes that can still take more buses
            break
        lines = [ln for ln in sc.lines if ln.name == name]
        hs = [ln.headway_min[hour] for ln in lines if np.isfinite(ln.headway_min[hour])]
        if not hs or min(hs) * factor < min_headway:
            continue
        cost = _bus_cost(city, lines, factor, hour, sc)
        if cost > 0:
            out.append(Action(f"B_{_short(name)}_{len(out)}", "boost", cost, 0.0, route=name, factor=factor,
                              meta={"boardings_per_hour": board[name]}))
    return out


# ---------------------------------------------------------------- search
def _group(a: Action) -> str:
    """Actions in the same group interact strongly and are not combined in one step."""
    if a.kind == "line":
        return f"line:{a.cand.meta.get('corridor', a.aid)}"
    return f"{a.kind}:{a.route}"


def _eval_worker(args):
    actions, which = args
    ev = _WORKER["cheap"] if which == "cheap" else _WORKER["fast"]
    return ev.evaluate(apply_actions(_WORKER["city"], actions, _WORKER["base"])).objective


def _parallel(city, evs, jobs, n_jobs, base=None):
    global _WORKER
    _WORKER = {"city": city, "base": base, **evs}
    if n_jobs <= 1 or len(jobs) <= 1 or "fork" not in mp.get_all_start_methods():
        return [_eval_worker(j) for j in jobs]
    with mp.get_context("fork").Pool(min(n_jobs, len(jobs))) as pool:
        return pool.map(_eval_worker, jobs, chunksize=1)


def run_frontier(city: City, on_checkpoint=None, start: dict | None = None) -> dict:
    """`start` resumes from an earlier run: {"scenario", "buses", "rail_km", "steps"};
    the scenario becomes the starting network and its spending counts against the caps."""
    fc = city.cfg["frontier"]
    hour = int(fc.get("cost_hour", 8))
    inv = inventory(city, hour=hour)
    B = fc["bus_share"] * inv["bus_fleet_peak"]
    K = fc["rail_share"] * inv["rail_km"]
    log.info("inventory %s -> budget: %.0f extra buses, %.1f new rail km", inv, B, K)
    # screening may use a subset of hours; the shortlist and the recorded path use all of them
    cheap = Evaluator(city, hours=fc.get("screen_hours") or None, K=fc["cheap_K"], M=1,
                      n_origins=fc["cheap_origins"])
    fast = Evaluator(city, K=fc["fast_K"], M=fc["fast_M"], n_origins=fc["fast_origins"])
    target = fc.get("target_min")
    evs = {"cheap": cheap, "fast": fast}
    n_jobs = fc.get("n_jobs") or os.cpu_count() or 1

    selected: list[Action] = []
    start_sc = start["scenario"] if start else city.base
    used_b, used_k = (int(start["buses"]), float(start["rail_km"])) if start else (0, 0.0)
    sc = start_sc
    j_fast = fast.evaluate(sc).summary()
    j_cheap = cheap.evaluate(sc).objective
    if start and start.get("steps"):
        steps = list(start["steps"])
    else:
        steps = [{"step": 0, "action": "mevcut durum", "buses": 0, "rail_km": 0.0, "spend": 0.0,
                  "objective": j_fast["objective"], "mean_t_daily": j_fast["daily"]["mean_t"],
                  **{f"mean_t_{h}": v["mean_t"] for h, v in j_fast["hours"].items()}}]
    step0 = steps[-1]["step"]
    steps[-1] = {**steps[-1], "objective": j_fast["objective"], "mean_t_daily": j_fast["daily"]["mean_t"]}
    checkpoints = sorted(fc["checkpoints"])
    next_cp = 0
    pool: list[Action] = []
    taken = set()
    spend = (used_b / max(B, 1) + used_k / max(K, 1e-6)) / 2
    cp_done_at = spend if start else -1.0
    while next_cp < len(checkpoints) and checkpoints[next_cp] <= spend:
        next_cp += 1
    for step in range(step0 + 1, step0 + fc["max_steps"] + 1):
        if (step - step0 - 1) % fc["regen_every"] == 0:
            cands = generate_candidates(city, cheap, sc)
            for c in cands:  # ids restart at NB0/NM0 every generation: make them unique
                c.cid = f"{c.cid}g{step}"
            pool = [a for a in line_actions(city, cands, hour) if a.aid not in taken]
            log.info("step %d: %d new-line candidates", step, len(pool))
        actions = [a for a in pool if a.aid not in taken]
        actions += extension_actions(city, sc, step)
        actions += boost_actions(city, sc, cheap, hour, fc["boost_top"], fc["boost_factor"],
                                 city.cfg["optimization"]["min_headway_min"])
        actions = [a for a in actions if used_b + a.cost_buses <= B + 1e-6 and used_k + a.cost_km <= K + 1e-6]
        if not actions:
            log.info("no affordable action left (buses %d/%.0f, rail %.1f/%.1f km)", used_b, B, used_k, K)
            break
        js = _parallel(city, evs, [(selected + [a], "cheap") for a in actions], n_jobs, start_sc)
        cost = np.array([a.cost_buses / max(B, 1) + a.cost_km / max(K, 1e-6) for a in actions])
        ratio = (j_cheap - np.array(js)) / np.maximum(cost, 1e-3)
        order = [i for i in np.argsort(-ratio) if j_cheap - js[i] > 0][: fc["shortlist"]]
        if not order:
            log.info("step %d: no action improves the objective", step)
            break
        jf = _parallel(city, evs, [(selected + [actions[i]], "fast") for i in order], n_jobs, start_sc)
        gains = [(steps[-1]["objective"] - j) / max(cost[i], 1e-3) for i, j in zip(order, jf)]
        ranked = sorted(range(len(order)), key=lambda q: -gains[q])
        if gains[ranked[0]] <= 0:
            log.info("step %d: shortlisted actions do not improve on the larger sample", step)
            break
        # take up to `batch` good actions on different routes / corridors at once
        batch = [actions[order[ranked[0]]]]
        groups = {_group(batch[0])}
        for q in ranked[1:]:
            a2 = actions[order[q]]
            if len(batch) >= fc.get("batch", 1) or gains[q] <= 0:
                break
            nb = used_b + sum(a.cost_buses for a in batch) + a2.cost_buses
            nk = used_k + sum(a.cost_km for a in batch) + a2.cost_km
            if _group(a2) in groups or nb > B + 1e-6 or nk > K + 1e-6:
                continue
            batch.append(a2)
            groups.add(_group(a2))
        if len(batch) > 1:  # keep the combination only if it beats the single best action
            j_batch = fast.evaluate(apply_actions(city, selected + batch, start_sc)).objective
            if j_batch >= jf[ranked[0]]:
                batch = batch[:1]
        for a in batch:
            selected.append(a)
            taken.add(a.aid)
            used_b += a.cost_buses
            used_k += a.cost_km
        sc = apply_actions(city, selected, start_sc)
        j_cheap = cheap.evaluate(sc).objective if len(batch) > 1 else js[order[ranked[0]]]
        summ = fast.evaluate(sc).summary()
        spend = (used_b / max(B, 1) + used_k / max(K, 1e-6)) / 2
        label = " + ".join(a.label() for a in batch)
        steps.append({"step": step, "action": label, "kind": "+".join(a.kind for a in batch),
                      "aid": "+".join(a.aid for a in batch), "buses": used_b,
                      "rail_km": round(used_k, 2), "spend": spend, "objective": summ["objective"],
                      "mean_t_daily": summ["daily"]["mean_t"],
                      **{f"mean_t_{h}": v["mean_t"] for h, v in summ["hours"].items()}})
        log.info("step %d: %s | buses %d/%.0f, rail %.1f/%.1f km | J %.2f | daily mean %.1f min", step, label,
                 used_b, B, used_k, K, summ["objective"], summ["daily"]["mean_t"])
        while next_cp < len(checkpoints) and spend >= checkpoints[next_cp]:
            if on_checkpoint:
                on_checkpoint(checkpoints[next_cp], sc, list(selected), steps)
                cp_done_at = spend
            next_cp += 1
        if target is not None and summ["daily"]["mean_t"] <= target:
            confirmed = on_checkpoint("target", sc, list(selected), steps) if on_checkpoint else None
            if confirmed is None or confirmed <= target:
                cp_done_at = spend
            if fc.get("stop_at_target", True) and (confirmed is None or confirmed <= target):
                log.info("target %.0f min reached (daily mean %.1f on the full sample)", target,
                         confirmed if confirmed is not None else summ["daily"]["mean_t"])
                break
    else:
        log.info("step limit reached")
    if on_checkpoint and steps[-1]["step"] > step0 and cp_done_at < spend:
        on_checkpoint(spend, sc, list(selected), steps)
    return {"inventory": inv, "budget": {"extra_buses": B, "rail_km": K}, "steps": steps,
            "selected": selected, "scenario": sc}
