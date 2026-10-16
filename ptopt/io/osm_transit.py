"""Rail lines (metro, tram, suburban rail, funicular) from OSM route relations.

Open GTFS for rail is often out of date (İBB's rail feed is from 2024 and misses
newer lines such as M11 to Halkalı or the M3/M4/M5 extensions). OSM route
relations list the stops of every line in order and are kept current, so they
are a better source for *where* lines run. They carry no timetable, so
headways come from the relation's `interval` tag when present and otherwise
from per-mode defaults in the config.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..model import Line

log = logging.getLogger(__name__)

ROUTE_MODES = {"subway": "metro", "light_rail": "metro", "monorail": "metro", "tram": "tram",
               "train": "rail", "funicular": "funicular"}
STOP_ROLES = ("stop", "stop_entry_only", "stop_exit_only", "stop_0", "stop_1")
PLATFORM_ROLES = ("platform", "platform_entry_only", "platform_exit_only")


def _read_relations(pbf_path: Path) -> list[dict]:
    import osmium

    rels = []
    fp = (osmium.FileProcessor(str(pbf_path), osmium.osm.RELATION)
          .with_filter(osmium.filter.TagFilter(*[("route", r) for r in ROUTE_MODES])))
    for r in fp:
        tags = dict(r.tags)
        stops = [m.ref for m in r.members if m.type == "n" and m.role in STOP_ROLES]
        platforms = [(m.type, m.ref) for m in r.members if m.role in PLATFORM_ROLES]
        rels.append({"id": r.id, "tags": tags, "stops": stops, "platforms": platforms})
    return rels


def _resolve_locations(pbf_path: Path, node_ids: set[int], way_ids: set[int]) -> tuple[dict, dict]:
    import osmium

    nodes: dict[int, tuple[float, float]] = {}
    if node_ids:
        for n in (osmium.FileProcessor(str(pbf_path), osmium.osm.NODE)
                  .with_filter(osmium.filter.IdFilter(node_ids))):
            nodes[n.id] = (n.location.lon, n.location.lat)
    ways: dict[int, tuple[float, float]] = {}
    if way_ids:
        for w in (osmium.FileProcessor(str(pbf_path), osmium.osm.NODE | osmium.osm.WAY)
                  .with_locations()
                  .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
                  .with_filter(osmium.filter.IdFilter(way_ids))):
            pts = [(nd.location.lon, nd.location.lat) for nd in w.nodes if nd.location.valid()]
            if pts:
                ways[w.id] = tuple(np.mean(pts, axis=0))
    return nodes, ways


def read_rail_relations(pbf_path: Path, cache: Path) -> list[dict]:
    """Every rail route relation in the extract with stop coordinates, cached as JSON."""
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    rels = _read_relations(pbf_path)
    node_ids = {s for r in rels for s in r["stops"]} | {ref for r in rels for t, ref in r["platforms"] if t == "n"}
    way_ids = {ref for r in rels for t, ref in r["platforms"] if t == "w"}
    nodes, ways = _resolve_locations(pbf_path, node_ids, way_ids)
    out = []
    for r in rels:
        pts = [nodes[s] for s in r["stops"] if s in nodes]
        if len(pts) < 2:  # some relations only list platforms
            pts = [nodes.get(ref) if t == "n" else ways.get(ref) for t, ref in r["platforms"]]
            pts = [p for p in pts if p is not None]
        if len(pts) >= 2:
            out.append({"id": r["id"], "tags": r["tags"], "stops": [list(p) for p in pts]})
    log.info("osm rail: %d route relations with stops", len(out))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False))
    return out


def _interval_minutes(tag: str | None) -> float | None:
    if not tag:
        return None
    m = re.match(r"^\s*(\d+):(\d+)(?::(\d+))?\s*$", tag)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        return h * 60 + mi + s / 60
    try:
        return float(tag)
    except ValueError:
        return None


def headway_profile(peak: float, off: float, first: int = 6, last: int = 24,
                    peak_hours=(7, 8, 9, 17, 18, 19)) -> np.ndarray:
    h = np.full(24, np.nan)
    h[first:last] = off
    h[list(peak_hours)] = peak
    return h


def rail_lines(relations: list[dict], cfg_osm: dict, inside_fn, proj):
    """Turn cached relations into (stops DataFrame, list[Line]).

    `inside_fn(lon, lat) -> bool array` keeps lines that have at least two stops
    in the study area; stops outside it are dropped from the line."""
    include = re.compile(cfg_osm.get("include", "."), re.I)
    exclude = re.compile(cfg_osm.get("exclude", r"^\b$"), re.I)
    train_include = re.compile(cfg_osm.get("train_include", "."), re.I)
    defaults = cfg_osm["headways"]
    merge_m = float(cfg_osm.get("merge_stop_m", 60))
    stop_xy: list[tuple[float, float]] = []
    stop_names: list[str] = []
    lines: list[Line] = []
    for rel in relations:
        tags = rel["tags"]
        route = tags.get("route")
        mode = ROUTE_MODES.get(route)
        label = " ".join(tags.get(k, "") for k in ("network", "operator", "ref", "name"))
        if mode is None or not include.search(label) or exclude.search(label):
            continue
        if route == "train" and not train_include.search(label):
            continue
        lon = np.array([p[0] for p in rel["stops"]])
        lat = np.array([p[1] for p in rel["stops"]])
        ok = inside_fn(lon, lat)
        if ok.sum() < 2:
            continue
        lon, lat = lon[ok], lat[ok]
        x, y = proj.to_xy(lon, lat)
        # drop consecutive duplicates (stop node + platform of the same station)
        keep = np.concatenate([[True], np.hypot(np.diff(x), np.diff(y)) > merge_m])
        x, y = x[keep], y[keep]
        if len(x) < 2:
            continue
        idx = np.arange(len(stop_xy), len(stop_xy) + len(x))
        stop_xy += list(zip(x, y))
        ref = tags.get("ref") or tags.get("name", "?")
        stop_names += [f"{ref} #{i + 1}" for i in range(len(x))]
        peak, off = defaults.get(route, defaults.get(mode, [6, 10]))
        iv = _interval_minutes(tags.get("interval"))
        if iv:
            peak, off = iv, max(iv, off)
        lines.append(Line(line_id=f"osm:{rel['id']}", name=f"osm:{ref}", mode=mode, stops=idx,
                          headway_min=headway_profile(peak, off),
                          meta={"osm_relation": rel["id"], "long_name": tags.get("name", "")}))
    xy = np.array(stop_xy) if stop_xy else np.zeros((0, 2))
    lo, la = proj.to_lonlat(xy[:, 0], xy[:, 1]) if len(xy) else (np.zeros(0), np.zeros(0))
    stops = pd.DataFrame({"stop_id": [f"osm:{i}" for i in range(len(xy))], "name": stop_names,
                          "x": xy[:, 0], "y": xy[:, 1], "lon": lo, "lat": la})
    log.info("osm rail: %d lines in study area (%s)", len(lines),
             ", ".join(sorted({ln.name.split(':', 1)[1] for ln in lines})))
    return stops, lines
