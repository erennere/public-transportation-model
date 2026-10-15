"""Drivable road network from an OSM .pbf extract, for routing bus lines on real streets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

DRIVABLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "living_street", "motorway_link", "trunk_link", "primary_link", "secondary_link",
            "tertiary_link"}
NO_ACCESS = {"no", "private"}
# routing cost per metre relative to a main road: buses keep to arterials and
# only use neighbourhood streets when they have to
CLASS_FACTOR = {"motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 1.1, "tertiary": 1.3,
                "unclassified": 2.0, "residential": 2.5, "living_street": 4.0}
MAJOR = {"primary", "secondary", "tertiary"}  # snapping targets: arterials with stops, not motorways
CACHE_VERSION = 3


def _class(hw: str) -> str:
    return hw[:-5] if hw.endswith("_link") else hw


@dataclass
class RoadGraph:
    x: np.ndarray
    y: np.ndarray
    G: csr_matrix  # directed; edge weight = length (m) x road-class factor
    major: np.ndarray | None = None  # node lies on a tertiary-or-better road

    def __post_init__(self):
        self.tree = cKDTree(np.column_stack([self.x, self.y]))
        if self.major is None:
            self.major = np.ones(len(self.x), bool)
        self._major_idx = np.flatnonzero(self.major)
        self.major_tree = cKDTree(np.column_stack([self.x[self._major_idx], self.y[self._major_idx]]))

    def __getstate__(self):
        return {"x": self.x, "y": self.y, "G": self.G, "major": self.major}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()

    def nearest(self, x, y, major: bool = False) -> np.ndarray:
        pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        if major and len(self._major_idx):
            return self._major_idx[self.major_tree.query(pts)[1]]
        return self.tree.query(pts)[1]

    def route_nodes(self, a: int, b: int) -> list[int]:
        dist, pred = dijkstra(self.G, directed=True, indices=a, return_predecessors=True)
        if a != b and pred[b] < 0:
            return []
        path = [b]
        while path[-1] != a:
            path.append(pred[path[-1]])
        return path[::-1]

    def route_via(self, xs, ys, major: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Road polyline visiting the given waypoints in order (empty if unroutable).
        With major=True, intermediate waypoints snap to the nearest main road."""
        nodes = self.nearest(xs, ys)
        if major and len(nodes) > 2:
            nodes[1:-1] = self.nearest(np.asarray(xs)[1:-1], np.asarray(ys)[1:-1], major=True)
        nodes = nodes[np.concatenate([[True], nodes[1:] != nodes[:-1]])]
        full: list[int] = [int(nodes[0])]
        for a, b in zip(nodes[:-1], nodes[1:]):
            seg = self.route_nodes(int(a), int(b))
            if not seg:
                return np.zeros(0), np.zeros(0)
            full += seg[1:]
        idx = np.array(full)
        return self.x[idx], self.y[idx]

    def route_corridor(self, cx, cy, width_m: float = 750.0) -> tuple[np.ndarray, np.ndarray]:
        """Street route from the start to the end of a corridor polyline that stays
        close to it: edge costs grow with (distance from corridor / width)^2, so the
        line follows main roads inside the corridor instead of being forced through
        fixed waypoints (which causes loops on one-way streets and motorways)."""
        from ..geo import resample_polyline

        px, py = resample_polyline(np.asarray(cx, float), np.asarray(cy, float), 100.0)
        d, _ = cKDTree(np.column_stack([px, py])).query(np.column_stack([self.x, self.y]))
        factor = 1.0 + (d / width_m) ** 2
        rows = np.repeat(np.arange(self.G.shape[0]), np.diff(self.G.indptr))
        G = csr_matrix((self.G.data * factor[rows], self.G.indices, self.G.indptr), shape=self.G.shape)
        a = int(self.nearest(px[0], py[0], major=True)[0])
        b = int(self.nearest(px[-1], py[-1], major=True)[0])
        _, pred = dijkstra(G, directed=True, indices=a, return_predecessors=True)
        if a == b or pred[b] < 0:
            return np.zeros(0), np.zeros(0)
        path = [b]
        while path[-1] != a:
            path.append(pred[path[-1]])
        idx = np.array(path[::-1])
        return self.x[idx], self.y[idx]

    def save(self, path: Path) -> None:
        g = self.G.tocoo()
        np.savez_compressed(path, x=self.x, y=self.y, r=g.row, c=g.col, w=g.data, major=self.major)

    @classmethod
    def load(cls, path: Path) -> "RoadGraph":
        d = np.load(path)
        n = len(d["x"])
        return cls(d["x"], d["y"], coo_matrix((d["w"], (d["r"], d["c"])), shape=(n, n)).tocsr(), d["major"])


def build_road_graph(pbf_path: Path, bbox, proj, pad_deg: float = 0.02) -> RoadGraph:
    """Read drivable ways inside bbox and keep the largest strongly connected part
    (so every kept node can reach every other one, respecting one-way streets)."""
    import osmium

    min_lon, min_lat, max_lon, max_lat = bbox
    min_lon, min_lat, max_lon, max_lat = min_lon - pad_deg, min_lat - pad_deg, max_lon + pad_deg, max_lat + pad_deg
    node_index: dict[int, int] = {}
    lons: list[float] = []
    lats: list[float] = []
    rows: list[int] = []
    cols: list[int] = []
    oneway_flags: list[int] = []
    factors: list[float] = []
    major_nodes: set[int] = set()

    fp = (osmium.FileProcessor(str(pbf_path), osmium.osm.NODE | osmium.osm.WAY)
          .with_locations()
          .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
          .with_filter(osmium.filter.KeyFilter("highway")))
    for w in fp:
        hw = w.tags.get("highway")
        if hw not in DRIVABLE or w.tags.get("access") in NO_ACCESS or w.tags.get("motor_vehicle") in NO_ACCESS:
            continue
        cls_ = _class(hw)
        factor = CLASS_FACTOR[cls_]
        ow = w.tags.get("oneway")
        flag = -1 if ow == "-1" else (1 if ow in ("yes", "true", "1") or hw.startswith("motorway") and ow != "no" else 0)
        prev = None
        for n in w.nodes:
            loc = n.location
            if not loc.valid():
                prev = None
                continue
            inside = min_lon <= loc.lon <= max_lon and min_lat <= loc.lat <= max_lat
            if not inside:
                prev = None
                continue
            i = node_index.get(n.ref)
            if i is None:
                i = len(lons)
                node_index[n.ref] = i
                lons.append(loc.lon)
                lats.append(loc.lat)
            if cls_ in MAJOR:
                major_nodes.add(i)
            if prev is not None and prev != i:
                rows.append(prev)
                cols.append(i)
                oneway_flags.append(flag)
                factors.append(factor)
            prev = i
    log.info("roads: %d nodes, %d segments read", len(lons), len(rows))
    x, y = proj.to_xy(np.array(lons), np.array(lats))
    r, c, f = np.array(rows), np.array(cols), np.array(oneway_flags)
    length = np.hypot(x[r] - x[c], y[r] - y[c]) * np.array(factors)
    fwd = f >= 0
    bwd = f <= 0
    rr = np.concatenate([r[fwd], c[bwd]])
    cc = np.concatenate([c[fwd], r[bwd]])
    ww = np.maximum(np.concatenate([length[fwd], length[bwd]]), 0.1)
    n = len(x)
    G = coo_matrix((ww, (rr, cc)), shape=(n, n)).tocsr()
    G.sum_duplicates()  # NB: duplicate parallel segments are rare; summing only raises their cost
    _, labels = connected_components(G, directed=True, connection="strong")
    keep = labels == np.bincount(labels).argmax()
    idx = np.flatnonzero(keep)
    G = G[idx][:, idx].tocsr()
    major = np.zeros(n, bool)
    major[list(major_nodes)] = True
    log.info("roads: %d nodes in largest strongly connected component (%d on main roads)",
             len(idx), int(major[idx].sum()))
    return RoadGraph(x[idx], y[idx], G, major[idx])


def load_or_build(pbf_path: Path, bbox, proj, cache: Path) -> RoadGraph:
    cache = Path(cache)
    if cache.exists():
        return RoadGraph.load(cache)
    g = build_road_graph(pbf_path, bbox, proj)
    cache.parent.mkdir(parents=True, exist_ok=True)
    g.save(cache)
    return g
