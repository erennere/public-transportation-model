import numpy as np
import pytest
from scipy.sparse import coo_matrix

from ptopt.candidates import bus_on_roads, default_headways, materialize, path_cumlen
from ptopt.config import load_config
from ptopt.io.roads import RoadGraph
from ptopt.model import Line
from ptopt.network import line_segment_minutes
from ptopt.optimize import Optimizer
from ptopt.traffic import ProfileTraffic
from tests.test_core import _toy_city


def _l_shaped_roads():
    """Streets along y=0 from x=0..2000 and then up x=2000 to y=2000, 100 m grid,
    plus a lone diagonal shortcut that is one-way in the wrong direction."""
    pts = [(x, 0.0) for x in range(0, 2001, 100)] + [(2000.0, y) for y in range(100, 2001, 100)]
    xy = np.array(pts, float)
    n = len(xy)
    r = np.arange(n - 1)
    c = r + 1
    w = np.hypot(*(xy[c] - xy[r]).T)
    rows, cols, ws = [r, c], [c, r], [w, w]
    # one-way shortcut from the far end back to the origin only
    rows.append(np.array([n - 1]))
    cols.append(np.array([0]))
    ws.append(np.array([np.hypot(2000, 2000)]))
    G = coo_matrix((np.concatenate(ws), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    return RoadGraph(xy[:, 0], xy[:, 1], G)


def test_route_follows_streets_and_respects_oneway():
    g = _l_shaped_roads()
    xs, ys = g.route_via([0, 2000], [0, 2000])
    assert path_cumlen(xs, ys)[-1] == pytest.approx(4000)  # around the corner, not the shortcut
    xs, ys = g.route_via([2000, 0], [2000, 0])
    assert path_cumlen(xs, ys)[-1] == pytest.approx(np.hypot(2000, 2000))  # shortcut allowed this way


def test_bus_candidate_stops_lie_on_streets():
    g = _l_shaped_roads()
    c = bus_on_roads("NB0", g, np.array([0.0, 2000.0]), np.array([0.0, 2000.0]), 400, 1500,
                     default_headways(10, 20), {})
    assert c is not None and c.on_roads
    d, _ = g.tree.query(np.column_stack([c.xs, c.ys]))
    assert d.max() <= 50  # every stop on (or right next to) a street node
    on_street = (np.isclose(c.ys, 0) & (c.xs <= 2000)) | np.isclose(c.xs, 2000)
    assert on_street.all()
    assert c.length_km == pytest.approx(4.0)
    assert c.seg_dist_m.sum() == pytest.approx(4000)


def test_segment_time_uses_street_distance():
    cfg = load_config(overrides={"network": {"dwell_s": 0.0, "bus_max_kmh": 100.0, "bus_speed_factor": 1.0}})
    traffic = ProfileTraffic(30.0, np.ones(24))
    ln = Line("L", "L", "bus", np.array([0, 1]), np.full(24, 10.0), meta={"seg_dist_m": np.array([3000.0])})
    t = line_segment_minutes(ln, np.array([0.0, 1000.0]), np.array([0.0, 0.0]), traffic, 8, cfg["network"])
    assert t[0] == pytest.approx(6.0)  # 3 km at 30 km/h, not the 1 km straight line


def test_materialize_carries_street_shape_and_distances():
    city = _toy_city()
    g = _l_shaped_roads()
    c = bus_on_roads("NB0", g, np.array([0.0, 2000.0]), np.array([0.0, 2000.0]), 400, 1500,
                     default_headways(10, 20), {})
    sc = materialize(city.base, [c], snap_m=0)
    a, b = sc.new_lines
    assert a.meta["seg_dist_m"].sum() == pytest.approx(4000)
    assert np.allclose(a.meta["seg_dist_m"], b.meta["seg_dist_m"][::-1])
    assert len(a.meta["shape"][0]) > len(a.stops) or len(a.meta["shape"][0]) >= 3


def test_shift_move_keeps_stops_on_street():
    city = _toy_city()
    city.roads = _l_shaped_roads()
    c = bus_on_roads("NB0", city.roads, np.array([0.0, 2000.0]), np.array([0.0, 2000.0]), 400, 1500,
                     default_headways(10, 20), {})
    opt = Optimizer.__new__(Optimizer)
    opt.city, opt.rng = city, np.random.default_rng(3)
    for kind in ["shift"] * 20 + ["trim", "extend"]:
        before = len(c.stop_s)
        opt._move_on_roads(c, kind, 400)
        assert np.all(np.diff(c.stop_s) >= 0)
        on_street = (np.isclose(c.ys, 0, atol=1e-6) & (c.xs <= 2000 + 1e-6)) | np.isclose(c.xs, 2000, atol=1e-6) \
            | np.isclose(c.xs, c.ys, atol=1e-6)  # the diagonal is a street too
        assert on_street.all(), kind
    assert len(c.stop_s) >= before - 1
