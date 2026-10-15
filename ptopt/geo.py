"""Geometry helpers: projection, distances, geohash decoding."""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer

_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
_GEOHASH_MAP = {c: i for i, c in enumerate(_GEOHASH_ALPHABET)}


def utm_crs_for(lon: float, lat: float) -> CRS:
    """Pick the UTM zone CRS that contains (lon, lat)."""
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


class Projector:
    """Converts WGS84 lon/lat <-> a metric CRS (meters)."""

    def __init__(self, crs: CRS | str | int):
        self.crs = CRS.from_user_input(crs)
        self._fwd = Transformer.from_crs(4326, self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, 4326, always_xy=True)

    def __getstate__(self):
        return {"crs": self.crs.to_wkt()}

    def __setstate__(self, state):
        self.__init__(state["crs"])

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


def euclid(x1, y1, x2, y2):
    return np.hypot(np.asarray(x2) - np.asarray(x1), np.asarray(y2) - np.asarray(y1))


def geohash_decode(gh: str) -> tuple[float, float]:
    """Return (lat, lon) of the centre of a geohash cell."""
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for ch in gh.strip().lower():
        v = _GEOHASH_MAP[ch]
        for bit in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2
                if v & bit:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if v & bit:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2


def polyline_length(xs, ys) -> float:
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 2:
        return 0.0
    return float(np.hypot(np.diff(xs), np.diff(ys)).sum())


def resample_polyline(xs, ys, spacing: float):
    """Points every `spacing` meters along a polyline (endpoints included)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total == 0:
        return xs[:1], ys[:1]
    n = max(1, int(round(total / spacing)))
    d = np.linspace(0, total, n + 1)
    return np.interp(d, cum, xs), np.interp(d, cum, ys)
