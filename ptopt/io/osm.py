"""Workplace / activity proxies from OpenStreetMap via Overpass."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import requests

log = logging.getLogger(__name__)

AMENITY_JOBS = ("school|university|college|hospital|clinic|doctors|bank|townhall|courthouse|"
                "police|marketplace|restaurant|cafe|fast_food|pharmacy|kindergarten|post_office|"
                "library|cinema|theatre|bus_station|fuel")
QUERY = """[out:json][timeout:600];
(
  nwr["office"]({b});
  nwr["shop"]({b});
  nwr["craft"]({b});
  nwr["amenity"~"^({amen})$"]({b});
  nwr["landuse"~"^(industrial|commercial|retail)$"]({b});
  nwr["building"~"^(office|commercial|industrial|retail|warehouse|hospital|university)$"]({b});
);
out center tags;"""


def _classify(tags: dict) -> str | None:
    if "office" in tags or tags.get("building") in ("office", "commercial"):
        return "office"
    if tags.get("landuse") == "industrial" or tags.get("building") in ("industrial", "warehouse"):
        return "industrial"
    if "shop" in tags or tags.get("landuse") in ("retail", "commercial") or tags.get("building") == "retail":
        return "shop"
    if "craft" in tags:
        return "craft"
    if "amenity" in tags or tags.get("building") in ("hospital", "university"):
        return "amenity"
    return None


def pois_from_pbf(pbf_path: Path, bbox, cache: Path) -> list[dict]:
    """Same POI classes as the Overpass query, read from a local .osm.pbf extract
    (e.g. BBBike or Geofabrik). Ways are reduced to the mean of their nodes."""
    import osmium

    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    min_lon, min_lat, max_lon, max_lat = bbox
    amen = set(AMENITY_JOBS.split("|"))
    keys = ("office", "shop", "craft", "amenity", "landuse", "building")
    out = []
    fp = (osmium.FileProcessor(str(pbf_path), osmium.osm.NODE | osmium.osm.WAY)
          .with_locations()
          .with_filter(osmium.filter.KeyFilter(*keys)))
    for obj in fp:
        tags = {k: obj.tags.get(k) for k in keys if k in obj.tags}
        if "amenity" in tags and tags["amenity"] not in amen:
            del tags["amenity"]
        if tags.get("landuse") not in (None, "industrial", "commercial", "retail"):
            del tags["landuse"]
        if tags.get("building") not in (None, "office", "commercial", "industrial", "retail", "warehouse",
                                        "hospital", "university"):
            del tags["building"]
        cls = _classify(tags)
        if cls is None:
            continue
        if obj.is_node():
            lon, lat = obj.location.lon, obj.location.lat
        else:
            pts = [(n.location.lon, n.location.lat) for n in obj.nodes if n.location.valid()]
            if not pts:
                continue
            lon, lat = np.mean(pts, axis=0)
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            out.append({"lon": float(lon), "lat": float(lat), "cls": cls})
    log.info("pbf: %d POIs in bbox", len(out))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def fetch_pois(bbox, overpass_url: str, cache: Path, tiles: tuple[int, int] = (4, 3),
               retries: int = 4) -> list[dict]:
    """Download POIs in bbox, tiled to keep Overpass responses small.
    Returns [{lon, lat, cls}] and caches them as JSON."""
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    min_lon, min_lat, max_lon, max_lat = bbox
    lons = np.linspace(min_lon, max_lon, tiles[0] + 1)
    lats = np.linspace(min_lat, max_lat, tiles[1] + 1)
    out, seen = [], set()
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            b = f"{lats[j]},{lons[i]},{lats[j + 1]},{lons[i + 1]}"
            q = QUERY.format(b=b, amen=AMENITY_JOBS)
            for attempt in range(retries):
                try:
                    r = requests.post(overpass_url, data={"data": q}, timeout=900)
                    r.raise_for_status()
                    elems = r.json()["elements"]
                    break
                except Exception as e:  # network hiccups, rate limits
                    wait = 10 * 2 ** attempt
                    log.warning("overpass tile %s failed (%s), retry in %ss", b, e, wait)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Overpass failed for tile {b}")
            for e in elems:
                key = (e["type"], e["id"])
                if key in seen:
                    continue
                seen.add(key)
                lat = e.get("lat", e.get("center", {}).get("lat"))
                lon = e.get("lon", e.get("center", {}).get("lon"))
                cls = _classify(e.get("tags", {}))
                if lat is None or cls is None:
                    continue
                out.append({"lon": lon, "lat": lat, "cls": cls})
            log.info("overpass tile %d,%d: %d elements (total %d)", i, j, len(elems), len(out))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out
