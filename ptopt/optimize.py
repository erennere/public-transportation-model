"""Choose new lines, adjust their stops, and re-allocate frequencies.

Stage 1  greedy: repeatedly add the candidate line with the best objective
         improvement while the budget (extra buses, metro km) allows.
Stage 2  simulated annealing over the selection: move a stop, extend or
         trim an end, change headway, swap / add / remove a line.
Stage 3  frequency re-allocation per time period with the square-root rule:
         for a fixed fleet, minimising total waiting gives
         vehicles_l ∝ sqrt(passengers_l * cycle_time_l).
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
from dataclasses import dataclass, field

import numpy as np

from .candidates import Candidate, cut_path, materialize, path_cumlen
from .model import City, Line, Scenario
from .network import line_cycle_minutes
from .simulate import Evaluator

log = logging.getLogger(__name__)
_WORKER_OPT = None  # set before forking worker processes


def _worker_objective(sel):
    return _WORKER_OPT.objective(sel)


@dataclass
class OptResult:
    selection: list[Candidate]
    scenario: Scenario
    objective: float
    base_objective: float
    history: list[dict] = field(default_factory=list)


class Optimizer:
    def __init__(self, city: City, ev: Evaluator, candidates: list[Candidate], rng=None):
        self.city = city
        self.cfg = city.cfg
        self.oc = city.cfg["optimization"]
        self.ev = ev
        self.cands = candidates
        self.rng = rng or np.random.default_rng(city.cfg["evaluation"]["seed"])
        self.snap = city.cfg["candidates"]["snap_existing_m"]
        self.history: list[dict] = []
        self._cache: dict = {}
        self.peak_hour = max(ev.hours, key=lambda h: city.hourly_profile[h])

    # ---------- cost / budget ----------
    def cost(self, sel: list[Candidate]) -> tuple[int, float]:
        buses, metro_km = 0, 0.0
        net = self.cfg["network"]
        for c in sel:
            if c.mode == "metro":
                metro_km += c.length_km
            else:
                ln = Line(c.cid, c.cid, c.mode, np.arange(len(c.xs)), c.headway_min,
                          meta={} if c.seg_dist_m is None else {"seg_dist_m": c.seg_dist_m})
                cyc = line_cycle_minutes(ln, c.xs, c.ys, self.city.traffic, self.peak_hour, net)
                buses += math.ceil(cyc / np.nanmin(c.headway_min))
        return buses, metro_km

    def feasible(self, sel) -> bool:
        b = self.oc["budget"]
        buses, km = self.cost(sel)
        return buses <= b["extra_buses"] and km <= b["metro_km"]

    # ---------- objective ----------
    def scenario(self, sel) -> Scenario:
        return materialize(self.city.base, sel, self.snap)

    @staticmethod
    def _key(sel):
        return tuple((c.cid, c.xs.round(0).tobytes(), c.ys.round(0).tobytes(),
                      np.nan_to_num(c.headway_min).tobytes()) for c in sel)

    def objective(self, sel) -> float:
        key = self._key(sel)
        if key not in self._cache:
            self._cache[key] = self.ev.evaluate(self.scenario(sel)).objective
        return self._cache[key]

    def objectives(self, sels: list[list[Candidate]]) -> list[float]:
        """Evaluate several selections, in parallel worker processes when possible."""
        n_jobs = self.oc.get("n_jobs") or os.cpu_count() or 1
        todo = [s for s in sels if self._key(s) not in self._cache]
        if n_jobs > 1 and len(todo) > 1 and "fork" in mp.get_all_start_methods():
            global _WORKER_OPT
            _WORKER_OPT = self
            with mp.get_context("fork").Pool(min(n_jobs, len(todo))) as pool:
                js = pool.map(_worker_objective, todo)
            for s, j in zip(todo, js):  # keep the parent cache warm
                self._cache[self._key(s)] = j
        return [self.objective(s) for s in sels]

    # ---------- stage 1 ----------
    def greedy(self, sel=None) -> list[Candidate]:
        sel = list(sel or [])
        best = self.objective(sel)
        for rnd in range(self.oc["greedy_rounds"]):
            used = {c.cid for c in sel}
            pool = [c for c in self.cands if c.cid not in used and self.feasible(sel + [c])]
            trial = None
            for c, j in zip(pool, self.objectives([sel + [c] for c in pool])):
                if j < best - 1e-9 and (trial is None or j < trial[0]):
                    trial = (j, c)
            if trial is None:
                break
            best, c = trial
            sel.append(c.copy())
            self.history.append({"stage": "greedy", "round": rnd, "added": c.cid, "objective": best})
            log.info("greedy round %d: +%s -> J=%.3f", rnd, c.cid, best)
        return sel

    # ---------- stage 2 ----------
    def _move(self, sel: list[Candidate]) -> list[Candidate] | None:
        rng = self.rng
        sel = [c.copy() for c in sel]
        kind = rng.choice(["shift", "shift", "extend", "trim", "headway", "swap", "add", "remove"])
        cc = self.cfg["candidates"]
        if kind in ("add", "swap") or not sel:
            pool = [c for c in self.cands if c.cid not in {s.cid for s in sel}]
            if not pool:
                return None
            new = pool[rng.integers(len(pool))].copy()
            if kind == "swap" and sel:
                sel[rng.integers(len(sel))] = new
            else:
                sel.append(new)
            return sel
        c = sel[rng.integers(len(sel))]
        spacing = cc["metro_station_spacing_m"] if c.mode == "metro" else cc["bus_stop_spacing_m"]
        if kind == "remove":
            sel.remove(c)
        elif c.on_roads and kind in ("shift", "extend", "trim"):
            if not self._move_on_roads(c, kind, spacing):
                return None
        elif kind == "shift":
            i = rng.integers(len(c.xs))
            c.xs[i] += rng.normal(0, spacing / 2)
            c.ys[i] += rng.normal(0, spacing / 2)
        elif kind == "extend":
            end = rng.integers(2)
            if end == 0:
                dx, dy = c.xs[0] - c.xs[1], c.ys[0] - c.ys[1]
            else:
                dx, dy = c.xs[-1] - c.xs[-2], c.ys[-1] - c.ys[-2]
            n = math.hypot(dx, dy) or 1.0
            ang = rng.normal(0, 0.5)
            ux, uy = dx / n, dy / n
            rx, ry = ux * math.cos(ang) - uy * math.sin(ang), ux * math.sin(ang) + uy * math.cos(ang)
            if end == 0:
                c.xs = np.insert(c.xs, 0, c.xs[0] + rx * spacing)
                c.ys = np.insert(c.ys, 0, c.ys[0] + ry * spacing)
            else:
                c.xs = np.append(c.xs, c.xs[-1] + rx * spacing)
                c.ys = np.append(c.ys, c.ys[-1] + ry * spacing)
        elif kind == "trim":
            if len(c.xs) <= 3:
                return None
            if rng.integers(2):
                c.xs, c.ys = c.xs[1:], c.ys[1:]
            else:
                c.xs, c.ys = c.xs[:-1], c.ys[:-1]
        elif kind == "headway":
            f = 0.8 if rng.integers(2) else 1.25
            c.headway_min = np.clip(c.headway_min * f, self.oc["min_headway_min"], self.oc["max_headway_min"])
        return sel

    def _move_on_roads(self, c: Candidate, kind: str, spacing: float) -> bool:
        """Stop moves that keep a road-routed bus line on the streets."""
        rng = self.rng
        s = c.stop_s
        if kind == "shift":  # slide one stop along the street, keeping stop order
            i = int(rng.integers(len(s)))
            lo = s[i - 1] + 50 if i > 0 else 0.0
            hi = s[i + 1] - 50 if i < len(s) - 1 else path_cumlen(c.path_xs, c.path_ys)[-1]
            if hi <= lo:
                return False
            s[i] = float(np.clip(s[i] + rng.normal(0, spacing / 2), lo, hi))
        elif kind == "trim":  # drop an end stop (and the street section beyond it)
            if len(s) <= 3:
                return False
            c.stop_s = s[1:] if rng.integers(2) else s[:-1]
        elif kind == "extend":  # route onward on the streets, roughly continuing the line
            at_start = bool(rng.integers(2))
            i0, i1 = (0, 1) if at_start else (-1, -2)
            dx, dy = c.xs[i0] - c.xs[i1], c.ys[i0] - c.ys[i1]
            n = math.hypot(dx, dy) or 1.0
            ang = rng.normal(0, 0.5)
            ux, uy = dx / n, dy / n
            tx = c.xs[i0] + 3 * spacing * (ux * math.cos(ang) - uy * math.sin(ang))
            ty = c.ys[i0] + 3 * spacing * (ux * math.sin(ang) + uy * math.cos(ang))
            if at_start:
                ex, ey = self.city.roads.route_via([tx, c.xs[0]], [ty, c.ys[0]])
            else:
                ex, ey = self.city.roads.route_via([c.xs[-1], tx], [c.ys[-1], ty])
            if len(ex) < 2:
                return False
            px, py = cut_path(c.path_xs, c.path_ys, s[0], s[-1])
            ext = path_cumlen(ex, ey)[-1]
            if ext < spacing / 2:
                return False
            base = s - s[0]
            if at_start:
                c.path_xs, c.path_ys = np.concatenate([ex, px[1:]]), np.concatenate([ey, py[1:]])
                mid = np.arange(ext - spacing, spacing / 2, -spacing)[::-1]
                c.stop_s = np.concatenate([[0.0], mid, base + ext])
            else:
                c.path_xs, c.path_ys = np.concatenate([px, ex[1:]]), np.concatenate([py, ey[1:]])
                mid = np.arange(spacing, ext - spacing / 2, spacing)
                c.stop_s = np.concatenate([base, base[-1] + mid, [base[-1] + ext]])
        c.place_stops()
        return True

    def anneal(self, sel: list[Candidate]) -> list[Candidate]:
        cur, j_cur = sel, self.objective(sel)
        best, j_best = cur, j_cur
        iters = self.oc["anneal_iters"]
        for it in range(iters):
            temp = self.oc["anneal_t0"] * (1 - it / iters) + 1e-6
            nxt = self._move(cur)
            if nxt is None or not self.feasible(nxt):
                continue
            j = self.objective(nxt)
            if j < j_cur or self.rng.random() < math.exp(-(j - j_cur) / temp):
                cur, j_cur = nxt, j
                if j < j_best:
                    best, j_best = nxt, j
                    self.history.append({"stage": "anneal", "iter": it, "objective": j_best,
                                         "lines": [c.cid for c in best]})
        log.info("anneal: J=%.3f", j_best)
        return best

    def run(self) -> OptResult:
        base_j = self.objective([])
        sel = self.greedy()
        if self.oc["anneal_iters"] > 0:
            sel = self.anneal(sel)
        return OptResult(sel, self.scenario(sel), self.objective(sel), base_j, self.history)


def optimize_frequencies(city: City, scenario: Scenario, K=None, M=None, log_cb=None) -> tuple[Scenario, list[dict]]:
    """Square-root-rule headway re-allocation per period with fixed fleet per
    mode. A period's change is kept only if it improves that period."""
    oc = city.cfg["optimization"]
    net = city.cfg["network"]
    sc = scenario.copy()
    report = []
    sx, sy = sc.stops["x"].to_numpy(), sc.stops["y"].to_numpy()
    for pname, hours in oc["periods"].items():
        rep_h = int(hours[len(hours) // 2])
        ev = Evaluator(city, hours=[rep_h], K=K or oc["eval_K"], M=M or oc["eval_M"])
        cur = ev.evaluate(sc, loads=True)
        j0 = cur.objective
        for _ in range(oc["frequency_iterations"]):
            loads: dict[str, float] = {}
            for s in cur.sims:
                for lid, (b, _) in s.loads.items():
                    loads[lid] = loads.get(lid, 0.0) + b / len(cur.sims)
            trial = sc.copy()
            # vehicles are only interchangeable within a pool (e.g. city buses);
            # metro / tram rolling stock is line specific
            active = [ln for ln in trial.lines
                      if np.isfinite(ln.headway_min[rep_h]) and ln.mode in oc["frequency_modes"]]
            for mode in {ln.mode for ln in active}:
                group = [ln for ln in active if ln.mode == mode]
                # each pattern is one direction: vehicles needed = one-way time (+layover) / headway
                P = np.array([loads.get(ln.line_id, 0.0) for ln in group])
                C = np.array([line_cycle_minutes(ln, sx, sy, city.traffic, rep_h, net) / 2 for ln in group])
                H = np.array([ln.headway_min[rep_h] for ln in group])
                fleet = (C / H).sum()
                score = np.sqrt(np.maximum(P, 1e-3) * C)
                h_target = C / np.maximum(fleet * score / score.sum(), 1e-6)
                h_new = np.clip(np.sqrt(H * h_target), oc["min_headway_min"], oc["max_headway_min"])
                # clipping changes the fleet; rescale so vehicles are conserved
                h_new *= (C / h_new).sum() / fleet
                for ln, hn, h0 in zip(group, h_new, H):
                    for h in hours:  # scale the whole period, keeping its hour-to-hour shape
                        if np.isfinite(ln.headway_min[h]):
                            ln.headway_min[h] *= hn / h0
            res = ev.evaluate(trial, loads=True)
            if res.objective < cur.objective:
                sc, cur = trial, res
            else:
                break
        report.append({"period": pname, "hour": rep_h, "objective_before": j0, "objective_after": cur.objective})
        if log_cb:
            log_cb(report[-1])
    return sc, report
