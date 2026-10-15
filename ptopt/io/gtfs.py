"""GTFS feeds -> stops + frequency-based line patterns.

Works with any GTFS feed. Includes repairs needed for the İBB open-data
exports (and harmless elsewhere):
  * ';' or ',' separators, UTF-8 / UTF-8-BOM / cp1254 encodings
  * double-encoded UTF-8 text ("KADIKÃ–Y" -> "KADIKÖY")
  * coordinates mangled by spreadsheet thousands separators
    ("410.191.700.005.564" -> 41.01917...), garbage suffixes
  * stop_times with times only at the first stop (times are then derived
    from distance and speed in the network model)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..model import GTFS_ROUTE_TYPES, Line

log = logging.getLogger(__name__)
_MOJIBAKE = re.compile("[ÃÄÅÂ]")
_NUM = re.compile(r"[-\d.,]+")


def read_table(path: Path, usecols=None) -> pd.DataFrame:
    raw = Path(path).read_bytes()[:20000]
    try:
        raw.decode("utf-8")
        enc = "utf-8-sig"
    except UnicodeDecodeError:
        # the sample may cut a multi-byte char; check a smaller prefix too
        try:
            raw[:-4].decode("utf-8")
            enc = "utf-8-sig"
        except UnicodeDecodeError:
            enc = "cp1254"
    header = raw.decode(enc if enc != "utf-8-sig" else "utf-8", errors="ignore").splitlines()[0]
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str, keep_default_na=False,
                     usecols=(lambda c: c.strip().lstrip("﻿") in usecols) if usecols else None)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]


def fix_text(s: str) -> str:
    if not isinstance(s, str) or not _MOJIBAKE.search(s):
        return s
    for enc in ("latin-1", "cp1252"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def repair_coord(raw: str, lo: float, hi: float) -> float:
    """Parse a coordinate, repairing lost decimal points so it falls in [lo, hi]."""
    if raw is None:
        return np.nan
    s = str(raw).strip()
    try:
        v = float(s.replace(",", ".")) if s.count(",") == 1 and "." not in s else float(s)
        if lo <= v <= hi:
            return v
    except ValueError:
        pass
    m = _NUM.match(s)
    if not m:
        return np.nan
    tok = m.group(0)
    try:
        v = float(tok)
        if lo <= v <= hi:
            return v
    except ValueError:
        pass
    neg = tok.startswith("-")
    digits = re.sub(r"\D", "", tok)
    for k in range(1, 4):
        if len(digits) <= k:
            break
        v = float(digits[:k] + "." + digits[k:]) * (-1 if neg else 1)
        if lo <= v <= hi:
            return v
    return np.nan


def _seconds(t: pd.Series) -> np.ndarray:
    parts = t.str.extract(r"^\s*(\d+):(\d+):?(\d*)")
    ok = parts[0].notna()
    out = np.full(len(t), np.nan)
    p = parts[ok]
    out[ok.to_numpy()] = (p[0].astype(int) * 3600 + p[1].astype(int) * 60
                          + p[2].replace("", "0").astype(int)).to_numpy()
    return out


def route_mode(route_type: str, short_name: str, overrides: dict) -> str:
    for pattern, mode in (overrides or {}).items():
        if re.fullmatch(pattern, str(short_name)):
            return mode
    try:
        rt = int(float(route_type))
    except ValueError:
        return "bus"
    if rt in GTFS_ROUTE_TYPES:
        return GTFS_ROUTE_TYPES[rt]
    # extended route types (Google): 100s rail, 400s metro, 700s bus, 900s tram, 1000/1200 ferry
    for lo, mode in ((100, "rail"), (400, "metro"), (700, "bus"), (900, "tram"), (1000, "ferry"),
                     (1200, "ferry"), (1300, "cable"), (1400, "funicular")):
        if lo <= rt < lo + 100:
            return mode
    return "bus"  # e.g. İBB minibus (9) and dolmuş (10)


def _weekday_services(cal: pd.DataFrame) -> set[str]:
    days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    n = sum(cal[d].astype(int) for d in days)
    return set(cal.loc[n >= 3, "service_id"])


def load_feed(feed: dict, base_dir: Path, bbox, proj, smooth_hours: int = 3):
    """Load one GTFS feed. Returns (stops DataFrame, list[Line])."""
    name = feed["name"]
    d = Path(base_dir) / feed.get("dir", name)
    min_lon, min_lat, max_lon, max_lat = bbox
    pad = 1.0
    excluded = set(str(x) for x in feed.get("exclude_route_types", []))

    stops = read_table(d / "stops.csv" if (d / "stops.csv").exists() else d / "stops.txt")
    stops["lat"] = [repair_coord(v, min_lat - pad, max_lat + pad) for v in stops["stop_lat"]]
    stops["lon"] = [repair_coord(v, min_lon - pad, max_lon + pad) for v in stops["stop_lon"]]
    bad = stops["lat"].isna() | stops["lon"].isna()
    if bad.any():
        log.warning("%s: %d stops with unusable coordinates dropped", name, int(bad.sum()))
    stops = stops[~bad]
    stops["inside"] = stops["lon"].between(min_lon, max_lon) & stops["lat"].between(min_lat, max_lat)
    stops["stop_name"] = stops["stop_name"].map(fix_text)
    stops = stops.set_index("stop_id")

    def table(stem, usecols=None):
        for ext in (".csv", ".txt"):
            if (d / (stem + ext)).exists():
                return read_table(d / (stem + ext), usecols)
        return None

    routes = table("routes")
    routes["route_short_name"] = routes["route_short_name"].map(fix_text)
    routes["route_long_name"] = routes.get("route_long_name", "").map(fix_text) if "route_long_name" in routes else ""
    routes = routes[~routes["route_type"].isin(excluded)]
    routes["mode"] = [route_mode(t, s, feed.get("mode_overrides")) for t, s in
                      zip(routes["route_type"], routes["route_short_name"])]
    routes = routes.set_index("route_id")

    trips = table("trips")
    cal = table("calendar")
    if cal is not None and len(cal):
        trips = trips[trips["service_id"].isin(_weekday_services(cal))]
    trips = trips[trips["route_id"].isin(routes.index)]
    trips = trips.set_index("trip_id")

    st = table("stop_times", usecols={"trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"})
    st = st[st["trip_id"].isin(trips.index)]
    st["seq"] = st["stop_sequence"].astype(int)
    st = st.sort_values(["trip_id", "seq"])
    dep = st["departure_time"].where(st["departure_time"] != "", st["arrival_time"])
    st["t"] = _seconds(dep)
    log.info("%s: %d weekday trips, %d stop_times rows", name, len(trips), len(st))

    g = st.groupby("trip_id", sort=False)
    seqs = g["stop_id"].agg(tuple)
    first_t = g["t"].first()
    full_times = g["t"].agg(lambda s: tuple(s) if s.notna().all() else None)

    freq = table("frequencies")
    departures: dict[str, np.ndarray] = {}
    if freq is not None and len(freq):
        freq = freq[freq["trip_id"].isin(seqs.index)]
        s0, s1 = _seconds(freq["start_time"]), _seconds(freq["end_time"])
        hw = freq["headway_secs"].astype(float).to_numpy()
        for tid, a, b, h in zip(freq["trip_id"], s0, s1, hw):
            if not (np.isfinite(a) and np.isfinite(b)) or h <= 0:
                continue
            times = np.arange(a, b, h)
            departures[tid] = np.concatenate([departures.get(tid, np.zeros(0)), times])

    # group trips into patterns (route, direction, exact stop sequence)
    route_of = trips["route_id"].to_dict()
    dir_of = trips["direction_id"].to_dict() if "direction_id" in trips else {}
    patterns: dict[tuple, list] = {}
    for tid, sq in seqs.items():
        patterns.setdefault((route_of[tid], dir_of.get(tid, ""), sq), []).append(tid)
    lines: list[Line] = []
    for k, (key, tids) in enumerate(patterns.items()):
        route_id, _dir, seq = key
        counts = np.zeros(24)
        for tid in tids:
            ts = departures.get(tid)
            if ts is None:
                ts = np.array([first_t[tid]])
            ts = ts[np.isfinite(ts)]
            np.add.at(counts, (ts // 3600).astype(int) % 24, 1)
        if counts.sum() == 0:
            continue
        if smooth_hours > 1:
            kern = np.ones(smooth_hours) / smooth_hours
            padded = np.concatenate([counts[-(smooth_hours // 2):], counts, counts[:smooth_hours // 2]])
            counts = np.convolve(padded, kern, mode="valid")
        with np.errstate(divide="ignore"):
            headway = np.where(counts > 0.05, 60.0 / counts, np.nan)
        seq = [s for s in seq if s in stops.index]
        inside = stops.loc[seq, "inside"].to_numpy() if seq else np.zeros(0, bool)
        # longest contiguous run of stops inside the study area
        best, cur_start, best_rng = 0, None, None
        for i, flag in enumerate(np.append(inside, False)):
            if flag and cur_start is None:
                cur_start = i
            elif not flag and cur_start is not None:
                if i - cur_start > best:
                    best, best_rng = i - cur_start, (cur_start, i)
                cur_start = None
        if best < 2:
            continue
        a, b = best_rng
        seg = None
        times = [full_times[t] for t in tids if full_times[t] is not None and len(full_times[t]) == len(seq)]
        if times:
            dt = np.median(np.diff(np.array(times, float), axis=1), axis=0)
            if np.all(dt >= 0):
                seg = dt[a:b - 1]
        r = routes.loc[route_id]
        lines.append(Line(line_id=f"{name}:{route_id}:{k}", name=f"{name}:{r['route_short_name']}",
                          mode=r["mode"], stops=np.array(seq[a:b], dtype=object), headway_min=headway,
                          seg_time_s=seg, meta={"route_id": route_id, "long_name": r.get("route_long_name", "")}))
    used = sorted({s for ln in lines for s in ln.stops})
    sub = stops.loc[used]
    x, y = proj.to_xy(sub["lon"].to_numpy(), sub["lat"].to_numpy())
    out = pd.DataFrame({"stop_id": [f"{name}:{s}" for s in used], "name": sub["stop_name"].to_numpy(),
                        "x": x, "y": y, "lon": sub["lon"].to_numpy(), "lat": sub["lat"].to_numpy()})
    pos = {s: i for i, s in enumerate(used)}
    for ln in lines:
        ln.stops = np.array([pos[s] for s in ln.stops], dtype=int)
    log.info("%s: %d stops, %d line patterns", name, len(out), len(lines))
    return out, lines


def load_feeds(feeds: list[dict], base_dir: Path, bbox, proj):
    all_stops, all_lines, offset = [], [], 0
    for feed in feeds:
        s, ls = load_feed(feed, base_dir, bbox, proj)
        for ln in ls:
            ln.stops = ln.stops + offset
        offset += len(s)
        all_stops.append(s)
        all_lines += ls
    stops = pd.concat(all_stops, ignore_index=True) if all_stops else pd.DataFrame(columns=["stop_id", "name", "x", "y"])
    return stops, all_lines
