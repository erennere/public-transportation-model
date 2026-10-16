"""Monte-Carlo evaluation of a scenario: M simulations x K sampled trips.

All scenarios are evaluated on the same sampled trips (common random
numbers), so differences between scenarios are not sampling noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .demand import Demand
from .model import City, Scenario
from .network import TransitGraph
from .sampling import ODSample, sample_od
from .traffic import car_time_min

UNREACHABLE_MIN = 180.0


@dataclass
class SimResult:
    hour: int
    mean_t: float
    share_le_z: float
    mean_excess: float
    car_share: float
    mean_car_t: float
    objective: float
    t_pt: np.ndarray | None = None
    loads: dict | None = None


METRICS = ("mean_t", "share_le_z", "mean_excess", "car_share", "mean_car_t", "objective")


@dataclass
class Evaluation:
    """Results of all simulations. Hours are combined with `hour_weights`
    (share of the evaluated hours' trips that travel in each hour), so the
    day-level numbers are passenger weighted: a busy 08:00 counts more than 12:00."""
    sims: list[SimResult]
    Z: float
    hour_weights: dict = field(default_factory=dict)

    def _weights(self) -> dict:
        hours = sorted({s.hour for s in self.sims})
        w = {h: float(self.hour_weights.get(h, 1.0)) for h in hours}
        tot = sum(w.values()) or 1.0
        return {h: v / tot for h, v in w.items()}

    def _hour_mean(self, key: str) -> dict:
        out = {}
        for h in {s.hour for s in self.sims}:
            out[h] = float(np.mean([getattr(s, key) for s in self.sims if s.hour == h]))
        return out

    def daily(self, key: str = "mean_t") -> float:
        w = self._weights()
        m = self._hour_mean(key)
        return float(sum(w[h] * m[h] for h in w))

    @property
    def objective(self) -> float:
        return self.daily("objective")

    def summary(self) -> dict:
        out = {"objective": self.objective, "Z_min": self.Z, "hour_weights": self._weights(),
               "daily": {k: self.daily(k) for k in METRICS}, "hours": {}}
        for h in sorted({s.hour for s in self.sims}):
            ss = [s for s in self.sims if s.hour == h]
            row = {}
            for key in METRICS:
                v = np.array([getattr(s, key) for s in ss])
                row[key] = float(v.mean())
                row[key + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            out["hours"][h] = row
        return out


class Evaluator:
    def __init__(self, city: City, hours=None, K=None, M=None, n_origins=None, seed=None):
        ev = city.cfg["evaluation"]
        self.city = city
        self.cfg = city.cfg
        self.hours = list(hours if hours is not None else ev["hours"])
        self.K = int(K or ev["K"])
        self.M = int(M or ev["M"])
        self.n_origins = int(n_origins or ev["origins_per_sim"])
        self.Z = float(ev["Z_min"])
        self.excess_w = float(ev["excess_weight"])
        seed = ev["seed"] if seed is None else seed
        self.demand = Demand(city.zones, city.cfg, city.hourly_profile)
        # passenger weight of each evaluated hour (trips starting in that hour)
        self.hour_weights = {h: float(self.demand.profile[h % 24]) for h in self.hours}
        z = city.zones
        net = city.cfg["network"]
        walk_m_per_min = net["walk_kmh"] * 1000.0 / 60.0
        self.samples: dict[int, list[ODSample]] = {}
        self.car_t: dict[int, list[np.ndarray]] = {}
        self.walk_t: dict[int, list[np.ndarray]] = {}
        for h in self.hours:
            T = self.demand.matrix(h)
            ss, cars, walks = [], [], []
            for m in range(self.M):
                rng = np.random.default_rng([seed, h, m])
                s = sample_od(T, self.K, self.n_origins, rng, ev["weighting"], ev["alpha"], hour=h)
                ss.append(s)
                cars.append(car_time_min(city.traffic, z.x[s.o], z.y[s.o], z.x[s.d], z.y[s.d], h, net))
                dist = np.hypot(z.x[s.o] - z.x[s.d], z.y[s.o] - z.y[s.d])
                walks.append(np.where(dist <= net["max_walk_only_m"],
                                      dist * net["walk_detour"] / walk_m_per_min, np.inf))
            self.samples[h], self.car_t[h], self.walk_t[h] = ss, cars, walks

    def evaluate(self, scenario: Scenario, keep_times: bool = False, loads: bool = False) -> Evaluation:
        mc = self.cfg["mode_choice"]
        sims = []
        for h in self.hours:
            g = TransitGraph(scenario, self.city.zones, self.city.traffic, self.cfg, h)
            # one shortest-path tree per distinct origin, shared by all simulations of this hour
            origins = np.unique(np.concatenate([s.o for s in self.samples[h]]))
            if loads:
                times, preds = g.shortest(origins, limit_min=UNREACHABLE_MIN, predecessors=True)
            else:
                times = g.shortest(origins, limit_min=UNREACHABLE_MIN)
            for m, s in enumerate(self.samples[h]):
                pos = np.searchsorted(origins, s.o)
                t_tr = times[pos, s.d]
                t = np.minimum(t_tr, self.walk_t[h][m])
                t = np.where(np.isfinite(t), t, UNREACHABLE_MIN)
                sims.append(self._metrics(h, t, self.car_t[h][m], s.weight, mc, keep_times))
                if loads:
                    b, pkm = g.line_loads(preds, pos, s.d, s.weight)
                    scale = self.demand.matrix(h).sum() / len(s.d)  # trips/hour per sample
                    sims[-1].loads = {ln.line_id: (b[i] * scale, pkm[i] * scale)
                                      for i, ln in enumerate(g.line_index)}
        return Evaluation(sims=sims, Z=self.Z, hour_weights=self.hour_weights)

    def _metrics(self, h, t, t_car, w, mc, keep) -> SimResult:
        wsum = w.sum()
        excess = np.maximum(0.0, t - self.Z)
        u = mc["beta_per_min"] * (t_car - t) + mc["asc_transit"]
        p_pt = 1.0 / (1.0 + np.exp(-u))
        car_share_if_avail = 1 - p_pt
        p_car = mc.get("car_availability", 1.0) * car_share_if_avail
        mean_t = float((w * t).sum() / wsum)
        mean_ex = float((w * excess).sum() / wsum)
        return SimResult(hour=h, mean_t=mean_t,
                         share_le_z=float((w * (t <= self.Z)).sum() / wsum),
                         mean_excess=mean_ex,
                         car_share=float((w * p_car).sum() / wsum),
                         mean_car_t=float((w * t_car).sum() / wsum),
                         objective=mean_t + self.excess_w * mean_ex,
                         t_pt=t if keep else None)


def corridor_times(city: City, scenario: Scenario, hours) -> list[dict]:
    """Door-to-door transit vs car time for user-defined X->Y corridors."""
    out = []
    cors = city.cfg["evaluation"].get("corridors") or []
    if not cors:
        return out
    z = city.zones
    for h in hours:
        g = TransitGraph(scenario, z, city.traffic, city.cfg, h)
        for c in cors:
            xx, xy = city.proj.to_xy(*c["x"])
            yx, yy = city.proj.to_xy(*c["y"])
            oi = int(np.argmin(np.hypot(z.x - xx, z.y - xy)))
            di = int(np.argmin(np.hypot(z.x - yx, z.y - yy)))
            t = float(g.shortest([oi], limit_min=UNREACHABLE_MIN)[0, di])
            car = float(car_time_min(city.traffic, [z.x[oi]], [z.y[oi]], [z.x[di]], [z.y[di]], h,
                                     city.cfg["network"])[0])
            out.append({"corridor": c.get("name", f"{c['x']}->{c['y']}"), "hour": h,
                        "transit_min": t if np.isfinite(t) else None, "car_min": car,
                        "meets_Z": bool(np.isfinite(t) and t <= city.cfg["evaluation"]["Z_min"])})
    return out
