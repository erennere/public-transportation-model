"""Administrative boundaries (e.g. a province) from an OSM .pbf extract."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import mapping, shape
from shapely.ops import transform

log = logging.getLogger(__name__)


def admin_boundary(pbf_path: Path, name: str, admin_level: int, cache: Path):
    """Polygon (WGS84 lon/lat) of the boundary=administrative area with the given
    name and admin_level. Cached as GeoJSON; building it reads the whole .pbf."""
    import osmium
    from shapely import wkb

    cache = Path(cache)
    if cache.exists():
        return shape(json.loads(cache.read_text()))
    fp = (osmium.FileProcessor(str(pbf_path), osmium.osm.NODE | osmium.osm.WAY | osmium.osm.RELATION
                               | osmium.osm.AREA)
          .with_areas(osmium.filter.TagFilter(("boundary", "administrative")))
          .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA))
          .with_filter(osmium.filter.TagFilter(("admin_level", str(admin_level)))))
    factory = osmium.geom.WKBFactory()
    geom = None
    for area in fp:
        if area.tags.get("name") == name:
            geom = wkb.loads(factory.create_multipolygon(area), hex=True)
            break
    if geom is None:
        raise ValueError(f"no admin_level={admin_level} boundary named {name!r} in {pbf_path}")
    log.info("boundary %s: bounds %s", name, np.round(geom.bounds, 3).tolist())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(mapping(geom)))
    return geom


def inside(geom_lonlat, proj, x, y) -> np.ndarray:
    """Which metric-CRS points (x, y) fall inside a lon/lat polygon."""
    g = transform(lambda lo, la, z=None: proj.to_xy(lo, la), geom_lonlat)
    return shapely.contains_xy(g, np.asarray(x, float), np.asarray(y, float))
