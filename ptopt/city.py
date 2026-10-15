"""Assemble a City (zones, transit network, traffic, demand profile) from config."""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from .geo import Projector, utm_crs_for
from .io.download import fetch, unzip_if_needed
from .io.gtfs import load_feeds
from .io.osm import fetch_pois, pois_from_pbf
from .io.ridership import ibb_hourly_profile
from .io.worldpop import read_points, worldpop_url
from .model import City, Scenario, Zones
from .synthetic import build_synthetic
from .traffic import GridTraffic, ProfileTraffic, ibb_hourly_speeds

log = logging.getLogger(__name__)


def data_paths(cfg: dict) -> tuple[Path, Path]:
    root = Path(cfg["data_dir"])
    return root / "raw", root / "processed"


def projector(cfg: dict) -> Projector:
    if cfg.get("crs"):
        return Projector(cfg["crs"])
    b = cfg["bbox"]
    return Projector(utm_crs_for((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))


def download_all(cfg: dict) -> None:
    """Fetch every raw input the config refers to (skips files already present)."""
    raw, _ = data_paths(cfg)
    pc = cfg["population"]
    if pc["source"] == "worldpop":
        url = pc.get("url") or worldpop_url(pc["iso3"], pc["year"])
        fetch(url, raw / "worldpop")
    jc = cfg["jobs"]
    if jc["source"] == "pbf":
        fetch(jc["pbf_url"], raw / "osm")
    get_pois(cfg)
    for feed in cfg["gtfs"]:
        for fname, url in (feed.get("urls") or {}).items():
            p = fetch(url, raw / feed.get("dir", feed["name"]), fname)
            unzip_if_needed(p)
    for section, sub in (("traffic", "traffic"), ("ridership", "ridership")):
        for url in cfg[section].get("urls") or []:
            fetch(url, raw / sub)


def get_pois(cfg: dict) -> list[dict]:
    raw, _ = data_paths(cfg)
    jc = cfg["jobs"]
    cache = raw / "osm" / f"pois_{jc['source']}_{_bbox_tag(cfg['bbox'])}.json"
    if jc["source"] == "pbf":
        return pois_from_pbf(raw / "osm" / Path(jc["pbf_url"]).name, cfg["bbox"], cache)
    if jc["source"] == "osm":
        return fetch_pois(cfg["bbox"], jc["overpass_url"], cache)
    return []


def _bbox_tag(b) -> str:
    return "_".join(f"{v:.3f}" for v in b)


def _grid(cfg, proj):
    b = cfg["bbox"]
    xs, ys = proj.to_xy([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def build_zones(cfg: dict, proj: Projector) -> Zones:
    raw, _ = data_paths(cfg)
    zc = cfg["zones"]
    cell = float(zc["cell_size_m"])
    x0, y0, x1, y1 = _grid(cfg, proj)
    nx = int(np.ceil((x1 - x0) / cell))
    ny = int(np.ceil((y1 - y0) / cell))

    def bin_points(lon, lat, w):
        x, y = proj.to_xy(lon, lat)
        i = np.floor((x - x0) / cell).astype(int)
        j = np.floor((y - y0) / cell).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        return np.bincount(j[ok] * nx + i[ok], weights=np.asarray(w, float)[ok], minlength=nx * ny)

    pc = cfg["population"]
    url = pc.get("url") or worldpop_url(pc["iso3"], pc["year"])
    tif = raw / "worldpop" / Path(url).name
    lon, lat, pop_v = read_points(tif, cfg["bbox"])
    pop = bin_points(lon, lat, pop_v)
    log.info("population in bbox: %.0f", pop.sum())

    jc = cfg["jobs"]
    pois = get_pois(cfg)
    w = np.array([jc["weights"].get(p["cls"], 1.0) for p in pois])
    jobs = bin_points([p["lon"] for p in pois], [p["lat"] for p in pois], w) if pois else np.zeros(nx * ny)
    jobs = jobs + jc["population_share"] * pop
    log.info("POIs: %d", len(pois))

    keep = (pop >= zc["min_population"]) | (jobs >= zc["min_jobs"])
    idx = np.flatnonzero(keep)
    cx = x0 + (idx % nx + 0.5) * cell
    cy = y0 + (idx // nx + 0.5) * cell
    return Zones(x=cx, y=cy, pop=pop[idx], jobs=jobs[idx], cell_size=cell)


def build_traffic(cfg: dict, proj: Projector):
    tc = cfg["traffic"]
    if tc["source"] == "ibb" and tc.get("urls"):
        raw, _ = data_paths(cfg)
        paths = [raw / "traffic" / Path(u).name for u in tc["urls"]]
        x, y, speeds = ibb_hourly_speeds(paths, proj, tc.get("weekdays_only", True))
        b = cfg["bbox"]
        bx0, by0, bx1, by1 = _grid(cfg, proj)
        pad = 3000
        ok = (x > bx0 - pad) & (x < bx1 + pad) & (y > by0 - pad) & (y < by1 + pad)
        log.info("traffic: %d geohash cells in area (bbox %s)", int(ok.sum()), b)
        return GridTraffic(x[ok], y[ok], speeds[ok], fallback_kmh=tc["free_flow_kmh"])
    return ProfileTraffic(tc["free_flow_kmh"], tc["hourly_factor"])


def build_profile(cfg: dict) -> np.ndarray:
    rc = cfg["ridership"]
    if rc["source"] == "ibb" and rc.get("urls"):
        raw, _ = data_paths(cfg)
        return ibb_hourly_profile([raw / "ridership" / Path(u).name for u in rc["urls"]])
    p = np.asarray(cfg["demand"]["hourly_profile"], float)
    return p / p.sum()


def _cache_key(cfg: dict) -> str:
    keys = ["bbox", "crs", "zones", "population", "jobs", "gtfs", "traffic", "ridership"]
    blob = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def load_city(cfg: dict, use_cache: bool = True) -> City:
    if cfg.get("synthetic") is not None:
        return build_synthetic(cfg)
    _, processed = data_paths(cfg)
    cache = processed / f"{cfg['name']}_{_cache_key(cfg)}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            city = pickle.load(f)
        city.cfg = cfg  # evaluation / optimisation settings may have changed
        return city
    proj = projector(cfg)
    zones = build_zones(cfg, proj)
    raw, _ = data_paths(cfg)
    stops, lines = load_feeds(cfg["gtfs"], raw, cfg["bbox"], proj)
    city = City(cfg=cfg, proj=proj, zones=zones, base=Scenario(stops, lines),
                traffic=build_traffic(cfg, proj), hourly_profile=build_profile(cfg))
    processed.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(city, f)
    return city
