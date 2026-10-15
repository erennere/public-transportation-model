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
