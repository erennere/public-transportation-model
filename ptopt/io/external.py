"""Travel times from commercial routing APIs (Google Routes, TomTom), for calibration.

Keys are read from the environment (GOOGLE_MAPS_API_KEY, TOMTOM_API_KEY) and never
written anywhere. Every answer is cached on disk, so a calibration can be
re-run without spending quota again. Only official APIs are used; map tiles
are not scraped (that is against the providers' terms).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

GOOGLE_MATRIX = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
TOMTOM_ROUTE = "https://api.tomtom.com/routing/1/calculateRoute/{o}:{d}/json"


def next_weekday_at(hour: int, utc_offset_h: int = 3, days_ahead_min: int = 1) -> dt.datetime:
    """Next Monday-Thursday at the given local hour (avoids Friday evening / weekend effects)."""
    tz = dt.timezone(dt.timedelta(hours=utc_offset_h))
    day = (dt.datetime.now(tz) + dt.timedelta(days=days_ahead_min)).date()
    while day.weekday() > 3:
        day += dt.timedelta(days=1)
    return dt.datetime(day.year, day.month, day.day, hour, 0, tzinfo=tz)


class Cache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(provider, mode, hour, o, d) -> str:
        return f"{provider}|{mode}|{hour}|{o[0]:.5f},{o[1]:.5f}|{d[0]:.5f},{d[1]:.5f}"

    def get(self, *k):
        return self.data.get(self.key(*k))

    def put(self, value, *k):
        self.data[self.key(*k)] = value

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))


def _latlng(p):
    return {"waypoint": {"location": {"latLng": {"latitude": p[1], "longitude": p[0]}}}}


def google_matrix(pairs, mode: str, hour: int, cache: Cache, key: str | None = None,
                  max_elements: int = 5000) -> list[float | None]:
    """Door-to-door minutes for (origin, destination) lon/lat pairs.
    mode: 'DRIVE' (traffic aware) or 'TRANSIT'. None where no route was found."""
    key = key or os.environ.get("GOOGLE_MAPS_API_KEY")
    out: list[float | None] = [None] * len(pairs)
    todo: dict[tuple, list[int]] = {}
    for i, (o, d) in enumerate(pairs):
        hit = cache.get("google", mode, hour, o, d)
        if hit is not None:
            out[i] = hit if hit >= 0 else None
        else:
            todo.setdefault(tuple(o), []).append(i)
    if not todo:
        return out
    if not key:
        log.warning("GOOGLE_MAPS_API_KEY not set: skipping %d Google %s requests", sum(map(len, todo.values())), mode)
        return out
    dep = next_weekday_at(hour).astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    per_request = 100 if mode == "TRANSIT" else 625
    sent = 0
    for o, idx in todo.items():
        for start in range(0, len(idx), per_request):
            chunk = idx[start:start + per_request]
            if sent + len(chunk) > max_elements:
                log.warning("google %s: element budget %d reached", mode, max_elements)
                cache.save()
                return out
            body = {"origins": [_latlng(o)], "destinations": [_latlng(pairs[i][1]) for i in chunk],
                    "travelMode": mode, "departureTime": dep}
            if mode == "DRIVE":
                body["routingPreference"] = "TRAFFIC_AWARE"
            for attempt in range(4):
                r = requests.post(GOOGLE_MATRIX, json=body, timeout=60, headers={
                    "X-Goog-Api-Key": key,
                    "X-Goog-FieldMask": "originIndex,destinationIndex,duration,condition,status"})
                if r.status_code in (429, 500, 503):
                    time.sleep(2 ** attempt)
                    continue
                break
            r.raise_for_status()
            sent += len(chunk)
            for el in r.json():
                i = chunk[el.get("destinationIndex", 0)]
                ok = el.get("condition") == "ROUTE_EXISTS" and "duration" in el
                minutes = float(el["duration"].rstrip("s")) / 60.0 if ok else -1.0
                cache.put(minutes, "google", mode, hour, pairs[i][0], pairs[i][1])
                out[i] = minutes if ok else None
    cache.save()
    log.info("google %s h%d: %d elements requested", mode, hour, sent)
    return out


def tomtom_drive(pairs, hour: int, cache: Cache, key: str | None = None,
                 max_requests: int = 2000, qps: float = 4.0) -> list[float | None]:
    """Car minutes with TomTom's historic traffic for a typical weekday at `hour`."""
    key = key or os.environ.get("TOMTOM_API_KEY")
    out: list[float | None] = [None] * len(pairs)
    dep = next_weekday_at(hour).isoformat()
    sent = 0
    for i, (o, d) in enumerate(pairs):
        hit = cache.get("tomtom", "DRIVE", hour, o, d)
        if hit is not None:
            out[i] = hit if hit >= 0 else None
            continue
        if not key:
            continue
        if sent >= max_requests:
            log.warning("tomtom: request budget %d reached", max_requests)
            break
        url = TOMTOM_ROUTE.format(o=f"{o[1]:.6f},{o[0]:.6f}", d=f"{d[1]:.6f},{d[0]:.6f}")
        for attempt in range(4):
            r = requests.get(url, timeout=60, params={"key": key, "traffic": "true", "departAt": dep,
                                                      "routeType": "fastest", "travelMode": "car"})
            if r.status_code in (429, 500, 503):
                time.sleep(2 ** attempt)
                continue
            break
        sent += 1
        time.sleep(1.0 / qps)
        if r.ok and r.json().get("routes"):
            minutes = r.json()["routes"][0]["summary"]["travelTimeInSeconds"] / 60.0
        else:
            minutes = -1.0
        cache.put(minutes, "tomtom", "DRIVE", hour, o, d)
        out[i] = minutes if minutes >= 0 else None
        if sent % 100 == 0:
            cache.save()
    if not key and any(v is None for v in out):
        log.warning("TOMTOM_API_KEY not set: TomTom times skipped")
    cache.save()
    log.info("tomtom DRIVE h%d: %d requests", hour, sent)
    return out
