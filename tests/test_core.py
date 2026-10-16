import numpy as np
import pandas as pd
import pytest

from ptopt.candidates import generate_candidates
from ptopt.config import load_config
from ptopt.demand import gravity
from ptopt.geo import Projector, geohash_decode
from ptopt.io.gtfs import fix_text, repair_coord, route_mode
from ptopt.model import City, Line, Scenario, Zones
from ptopt.network import TransitGraph
from ptopt.optimize import Optimizer
from ptopt.sampling import sample_od
from ptopt.simulate import Evaluator
from ptopt.synthetic import build_synthetic
from ptopt.traffic import ProfileTraffic


@pytest.fixture(scope="module")
def synth():
    cfg = load_config(overrides={"synthetic": {"half_size_km": 10},
                                 "evaluation": {"K": 400, "M": 2, "origins_per_sim": 60},
                                 "optimization": {"greedy_rounds": 2, "anneal_iters": 10}})
    return build_synthetic(cfg)


def test_geohash_decode():
    lat, lon = geohash_decode("sxkd1k")
    assert abs(lat - 41.1575) < 0.01 and abs(lon - 28.8885) < 0.01


@pytest.mark.parametrize("raw,lo,hi,expected", [
    ("410.191.700.005.564", 40, 42, 41.0191700005564),
    ("286.843.529.999.755", 27, 30, 28.6843529999755),
    ("41.010.675.000.555", 40, 42, 41.010675000555),
    ("41.00506785536913:17 17.03.20268", 40, 42, 41.00506785536913),
    ("41.083287780558", 40, 42, 41.083287780558),
    ("41,0833", 40, 42, 41.0833),
])
def test_repair_coord(raw, lo, hi, expected):
    assert repair_coord(raw, lo, hi) == pytest.approx(expected)


def test_repair_coord_garbage():
    assert np.isnan(repair_coord("abc", 40, 42))


def test_fix_text_mojibake():
    assert fix_text("KADIKÃ–Y - KÄ°RAZLITEPE") == "KADIKÖY - KİRAZLITEPE"
    assert fix_text("ÜSKÜDAR") == "ÜSKÜDAR"


def test_route_mode():
    assert route_mode("3", "34AS", {"34[A-Z]*": "brt"}) == "brt"
    assert route_mode("3", "15F", {"34[A-Z]*": "brt"}) == "bus"
    assert route_mode("1", "M4", {}) == "metro"
    assert route_mode("4", "x", {}) == "ferry"
    assert route_mode("715", "x", {}) == "bus"


def test_gravity_marginals():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 20, size=(30, 2))
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    o = rng.uniform(100, 1000, 30)
    dd = rng.uniform(100, 1000, 30)
    T = gravity(o, dd, d, 5.0, iterations=200)
    assert np.allclose(T.sum(1), o, rtol=1e-3)
    assert np.allclose(T.sum(0), dd * o.sum() / dd.sum(), rtol=1e-3)
    assert np.all(np.diag(T) == 0)


def test_sampling_follows_dominant_flow():
    # flows: X->Y strongest, Y->Z weaker, Z->X weakest
    T = np.zeros((3, 3))
    T[0, 1], T[1, 2], T[2, 0] = 600, 300, 100
    s = sample_od(T, 20000, 3000, np.random.default_rng(1))
    share = {(0, 1): np.mean((s.o == 0) & (s.d == 1)), (1, 2): np.mean((s.o == 1) & (s.d == 2)),
             (2, 0): np.mean((s.o == 2) & (s.d == 0))}
    assert share[(0, 1)] > 0.5 > share[(1, 2)] > share[(2, 0)]
    assert share[(0, 1)] == pytest.approx(0.6, abs=0.03)
    # tempered sampling keeps the order but flattens it; weights restore flow shares
    t = sample_od(T, 20000, 3000, np.random.default_rng(1), weighting="tempered", alpha=0.5)
    sh_t = np.mean((t.o == 0) & (t.d == 1))
    assert 0.33 < sh_t < 0.6
    assert np.mean(t.weight * ((t.o == 0) & (t.d == 1))) == pytest.approx(0.6, abs=0.04)


def _toy_city():
    """Two zones 3 km apart, one bus line between stops next to them."""
    cfg = load_config(overrides={"network": {"boarding_penalty_min": 0.0}})
    zones = Zones(x=np.array([0.0, 3000.0]), y=np.zeros(2), pop=np.ones(2), jobs=np.ones(2), cell_size=1000)
    stops = pd.DataFrame({"stop_id": ["a", "b"], "name": ["a", "b"], "x": [100.0, 2900.0], "y": [0.0, 0.0]})
    hw = np.full(24, 10.0)
    line = Line("L", "L", "metro", np.array([0, 1]), hw, seg_time_s=np.array([300.0]))
    traffic = ProfileTraffic(50, np.ones(24))
    return City(cfg, Projector(32635), zones, Scenario(stops, [line]), traffic, np.ones(24) / 24)


def test_transit_graph_time_matches_hand_calculation():
    city = _toy_city()
    g = TransitGraph(city.base, city.zones, city.traffic, city.cfg, hour=8)
    t = g.shortest([0])[0, 1]
    walk = 100 * 1.3 / (4.5 * 1000 / 60)  # access and egress
    expected = walk + 5.0 + 5.0 + walk  # wait (10/2) + ride (300 s) + egress
    assert t == pytest.approx(expected, abs=1e-3)


def test_no_service_hour_falls_back_to_walking_only():
    city = _toy_city()
    city.base.lines[0].headway_min[3] = np.nan
    g = TransitGraph(city.base, city.zones, city.traffic, city.cfg, hour=3)
    assert not np.isfinite(g.shortest([0])[0, 1])


def test_evaluator_runs_and_is_deterministic(synth):
    ev = Evaluator(synth)
    a = ev.evaluate(synth.base).objective
    b = ev.evaluate(synth.base).objective
    assert a == b and a > 0


def test_optimizer_improves_synthetic_city(synth):
    ev = Evaluator(synth)
    cands = generate_candidates(synth, ev)
    assert cands
    opt = Optimizer(synth, ev, cands)
    res = opt.run()
    assert res.objective < res.base_objective
    buses, km = opt.cost(res.selection)
    budget = synth.cfg["optimization"]["budget"]
    assert buses <= budget["extra_buses"] and km <= budget["metro_km"]


def test_merge_variants_adds_frequencies():
    from ptopt.io.gtfs import merge_variants
    hw = np.full(24, 30.0)
    trunk = Line("a", "iett:15F", "bus", np.array(list("ABCDEFGHIJ"), dtype=object), hw.copy(), meta={"direction": "0"})
    short = Line("b", "iett:15F", "bus", np.array(list("CDEFGH"), dtype=object), hw.copy(), meta={"direction": "0"})
    other_dir = Line("c", "iett:15F", "bus", np.array(list("JIHG"), dtype=object), hw.copy(), meta={"direction": "1"})
    detour = Line("d", "iett:15F", "bus", np.array(list("AXYZW"), dtype=object), hw.copy(), meta={"direction": "0"})
    out = merge_variants([trunk, short, other_dir, detour])
    assert len(out) == 3
    merged = next(l for l in out if l.line_id == "a")
    assert merged.headway_min[8] == pytest.approx(15.0)  # two 30-min variants -> every 15 min
    assert next(l for l in out if l.line_id == "d").headway_min[8] == pytest.approx(30.0)


def test_common_lines_halve_the_wait():
    city = _toy_city()
    city.cfg["network"]["max_wait_min"] = 60
    g1 = TransitGraph(city.base, city.zones, city.traffic, city.cfg, hour=8)
    t1 = g1.shortest([0])[0, 1]
    twin = city.base.lines[0].copy()
    twin.line_id = "L2"
    city.base.lines.append(twin)
    g2 = TransitGraph(city.base, city.zones, city.traffic, city.cfg, hour=8)
    t2 = g2.shortest([0])[0, 1]
    assert t1 - t2 == pytest.approx(5.0 - 2.5, abs=1e-3)  # 10-min headway -> combined 5 min


def test_car_calibration_recovers_known_slowdown(synth):
    from ptopt import calibrate
    from ptopt.traffic import car_time_min
    net = synth.cfg["network"]
    s = calibrate.sample_pairs(synth, 60, [8])
    slow = ProfileTraffic(synth.traffic.free_flow * 0.5, synth.traffic.factor, synth.traffic.centres)
    z = synth.zones
    s.car_obs = np.array([car_time_min(slow, [z.x[o]], [z.y[o]], [z.x[d]], [z.y[d]], 8, net)[0]
                          for o, d in zip(s.o, s.d)])
    traffic, car = calibrate.fit_car(synth, s)
    assert car["global_factor"] == pytest.approx(0.5, rel=0.05)
    assert car["stats"]["after"]["median_abs_pct_err"] < 3.0 < car["stats"]["before"]["median_abs_pct_err"]


def test_frontier_runs_within_budget_and_improves():
    from ptopt.frontier import inventory, run_frontier
    cfg = load_config(overrides={"synthetic": {"half_size_km": 10},
                                 "evaluation": {"K": 300, "M": 1, "origins_per_sim": 50},
                                 "candidates": {"n_corridors": 6},
                                 "frontier": {"cheap_K": 200, "cheap_origins": 40, "fast_K": 300, "fast_M": 1,
                                              "fast_origins": 50, "max_steps": 3, "shortlist": 3, "n_jobs": 1,
                                              "boost_top": 4, "checkpoints": [0.5, 1.0]}})
    city = build_synthetic(cfg)
    inv = inventory(city)
    assert inv["bus_fleet_peak"] > 0 and inv["rail_km"] > 0
    seen = []
    res = run_frontier(city, on_checkpoint=lambda lvl, sc, sel, steps: seen.append(lvl))
    steps = res["steps"]
    assert len(steps) >= 2
    assert steps[-1]["objective"] < steps[0]["objective"]
    assert steps[-1]["buses"] <= res["budget"]["extra_buses"] + 1e-6
    assert steps[-1]["rail_km"] <= res["budget"]["rail_km"] + 1e-6
    assert all(b["spend"] >= a["spend"] for a, b in zip(steps, steps[1:]))
    assert seen  # at least the final checkpoint
