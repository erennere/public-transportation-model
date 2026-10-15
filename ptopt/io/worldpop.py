"""WorldPop population rasters (R2025A, 100 m, constrained)."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import from_bounds

URL_TEMPLATE = ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{year}/{ISO}/v1/"
                "100m/constrained/{iso}_pop_{year}_CN_100m_R2025A_v1.tif")


def worldpop_url(iso3: str, year: int) -> str:
    return URL_TEMPLATE.format(year=year, ISO=iso3.upper(), iso=iso3.lower())


def read_points(path, bbox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel centres (lon, lat) and population inside bbox, populated pixels only."""
    min_lon, min_lat, max_lon, max_lat = bbox
    with rasterio.open(path) as src:
        win = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win, masked=True).filled(0).astype(float)
        tr = src.window_transform(win)
    arr[~np.isfinite(arr) | (arr < 0)] = 0
    r, c = np.nonzero(arr > 0)
    lon = tr.c + (c + 0.5) * tr.a
    lat = tr.f + (r + 0.5) * tr.e
    return lon, lat, arr[r, c]
