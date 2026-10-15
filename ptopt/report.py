"""Outputs: GeoJSON layers, metric tables and a quick-look PNG map."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import City, Scenario


def _fc(features):
    return {"type": "FeatureCollection", "features": features}


def scenario_geojson(city: City, sc: Scenario, only_new: bool = False) -> dict:
    lon, lat = city.proj.to_lonlat(sc.stops["x"].to_numpy(), sc.stops["y"].to_numpy())
    feats, seen = [], set()
    for ln in sc.lines:
        if only_new and not ln.is_new:
            continue
        if ln.name in seen:  # one feature per line (direction A)
            continue
        seen.add(ln.name)
        coords = [[float(lon[i]), float(lat[i])] for i in ln.stops]
        hw = {str(h): (None if not np.isfinite(v) else round(float(v), 1)) for h, v in enumerate(ln.headway_min)}
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                      "properties": {"line": ln.name, "mode": ln.mode, "new": ln.is_new, "headway_min": hw}})
        if ln.is_new:
            for k, i in enumerate(ln.stops):
                feats.append({"type": "Feature",
                              "geometry": {"type": "Point", "coordinates": [float(lon[i]), float(lat[i])]},
                              "properties": {"line": ln.name, "mode": ln.mode, "stop_seq": k,
                                             "stop_id": str(sc.stops["stop_id"].iat[i]),
                                             "name": str(sc.stops["name"].iat[i])}})
    return _fc(feats)


def zones_geojson(city: City) -> dict:
    z = city.zones
    h = z.cell_size / 2
    feats = []
    for i in range(len(z)):
        xs = [z.x[i] - h, z.x[i] + h, z.x[i] + h, z.x[i] - h, z.x[i] - h]
        ys = [z.y[i] - h, z.y[i] - h, z.y[i] + h, z.y[i] + h, z.y[i] - h]
        lo, la = city.proj.to_lonlat(xs, ys)
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [[[float(a), float(b)] for a, b in zip(lo, la)]]},
                      "properties": {"zone": i, "pop": round(float(z.pop[i])), "jobs": round(float(z.jobs[i]))}})
    return _fc(feats)


def plot_map(city: City, base: Scenario, opt: Scenario | None, path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = city.zones
    fig, ax = plt.subplots(figsize=(10, 9))
    from matplotlib.collections import PolyCollection

    h = z.cell_size / 2
    squares = [[(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)] for x, y in zip(z.x, z.y)]
    pc = PolyCollection(squares, array=z.pop, cmap="Greys", alpha=0.55, linewidths=0)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, shrink=0.6, label="population per zone")
    sx, sy = base.stops["x"].to_numpy(), base.stops["y"].to_numpy()
    styles = {"bus": ("#9aa7b8", 0.4), "brt": ("#d62728", 1.8), "metro": ("#1f77b4", 2.2),
              "rail": ("#1f77b4", 2.2), "tram": ("#2ca02c", 1.5), "ferry": ("#17becf", 1.0)}
    for ln in base.lines:
        col, lw = styles.get(ln.mode, ("#999999", 0.5))
        ax.plot(sx[ln.stops], sy[ln.stops], color=col, lw=lw, alpha=0.7 if ln.mode != "bus" else 0.35)
    if opt is not None:
        ox, oy = opt.stops["x"].to_numpy(), opt.stops["y"].to_numpy()
        seen = set()
        for ln in opt.new_lines:
            if ln.name in seen:
                continue
            seen.add(ln.name)
            col = "#ff7f0e" if ln.mode == "bus" else "#9467bd"
            ax.plot(ox[ln.stops], oy[ln.stops], color=col, lw=3, marker="o", ms=3.5, label=f"new {ln.mode} {ln.name}")
        if seen:
            ax.legend(loc="upper left", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def summary_markdown(base: dict, opt: dict | None, corridors_base, corridors_opt, selection_info) -> str:
    rows = ["| hour | scenario | mean time (min) | share ≤ Z | mean excess over Z (min) | car share | objective |",
            "|---|---|---|---|---|---|---|"]
    for h, b in base["hours"].items():
        rows.append(f"| {h}:00 | base | {b['mean_t']:.1f} ± {b['mean_t_sd']:.1f} | {b['share_le_z']:.1%} | "
                    f"{b['mean_excess']:.1f} | {b['car_share']:.1%} | {b['objective']:.1f} |")
        if opt:
            o = opt["hours"][h]
            rows.append(f"| {h}:00 | optimised | {o['mean_t']:.1f} ± {o['mean_t_sd']:.1f} | {o['share_le_z']:.1%} | "
                        f"{o['mean_excess']:.1f} | {o['car_share']:.1%} | {o['objective']:.1f} |")
    out = [f"Z = {base['Z_min']:.0f} min\n", "\n".join(rows)]
    if selection_info:
        out.append("\n**New lines**\n\n| line | mode | stops | length (km) | peak headway (min) |\n|---|---|---|---|---|")
        for s in selection_info:
            out.append(f"| {s['line']} | {s['mode']} | {s['stops']} | {s['length_km']:.1f} | {s['peak_headway']:.1f} |")
    if corridors_base:
        out.append("\n**Corridors**\n\n| corridor | hour | transit base | transit optimised | car |\n|---|---|---|---|---|")
        for a, b in zip(corridors_base, corridors_opt or corridors_base):
            f = lambda v: "—" if v is None else f"{v:.0f}"
            out.append(f"| {a['corridor']} | {a['hour']} | {f(a['transit_min'])} | {f(b['transit_min'])} | {a['car_min']:.0f} |")
    return "\n".join(out) + "\n"
