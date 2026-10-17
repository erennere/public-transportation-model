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
        if "shape" in ln.meta:  # street geometry of road-routed new lines
            slo, sla = city.proj.to_lonlat(*ln.meta["shape"])
            coords = [[float(a), float(b)] for a, b in zip(slo, sla)]
        else:
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


RAIL_MODES = ("metro", "rail", "tram", "funicular", "cable")
NEW_COLORS = ["#e6550d", "#7b3294", "#008837", "#d01c8b", "#2b83ba", "#a6611a", "#1b7837", "#c51b7d",
              "#5e3c99", "#e66101", "#018571", "#b2182b"]


def _short(name: str) -> str:
    return name.split(":", 1)[-1]


def _merc(city: City, x, y):
    from pyproj import Transformer

    lon, lat = city.proj.to_lonlat(x, y)
    t = Transformer.from_crs(4326, 3857, always_xy=True)
    return t.transform(lon, lat)


def plot_map(city: City, base: Scenario, opt: Scenario | None, path: Path, title: str = "") -> None:
    """Static map on the Esri World Street Map basemap (place names, roads, coastline)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    z = city.zones
    fig, ax = plt.subplots(figsize=(13, 10))
    h = z.cell_size / 2
    sq = []
    for x, y in zip(z.x, z.y):
        mx, my = _merc(city, np.array([x - h, x + h, x + h, x - h]), np.array([y - h, y - h, y + h, y + h]))
        sq.append(list(zip(mx, my)))
    dens = z.pop / (z.cell_size / 1000) ** 2
    pc = PolyCollection(sq, array=dens, cmap="YlOrRd", alpha=0.28, linewidths=0, zorder=2)
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, shrink=0.55, pad=0.01, label="nüfus yoğunluğu (kişi/km², WorldPop 2026)")

    bx, by = _merc(city, base.stops["x"].to_numpy(), base.stops["y"].to_numpy())
    labelled = set()
    for ln in base.lines:
        xs, ys = bx[ln.stops], by[ln.stops]
        if ln.mode == "bus":
            ax.plot(xs, ys, color="#7a8899", lw=0.4, alpha=0.25, zorder=3)
        elif ln.mode == "ferry":
            ax.plot(xs, ys, color="#3a9bd9", lw=0.8, ls="--", alpha=0.6, zorder=3)
        else:
            col = "#c0392b" if ln.mode == "brt" else "#1f4e9c"
            ax.plot(xs, ys, color=col, lw=3.0, alpha=0.9, zorder=4, solid_capstyle="round")
            ax.scatter(xs, ys, s=10, color="white", edgecolor=col, lw=1, zorder=5)
            nm = _short(ln.name)
            if nm not in labelled and len(xs) >= 2:
                labelled.add(nm)
                k = len(xs) // 2
                ax.annotate(nm, (xs[k], ys[k]), fontsize=8, fontweight="bold", color=col, zorder=7,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.6, alpha=0.9))
    handles = [Line2D([], [], color="#1f4e9c", lw=3, label="mevcut metro / Marmaray / tramvay"),
               Line2D([], [], color="#c0392b", lw=3, label="Metrobüs"),
               Line2D([], [], color="#7a8899", lw=1, label="mevcut İETT otobüs hatları"),
               Line2D([], [], color="#3a9bd9", lw=1, ls="--", label="vapur")]
    if opt is not None:
        ox, oy = _merc(city, opt.stops["x"].to_numpy(), opt.stops["y"].to_numpy())
        uniq = list({ln.name: ln for ln in opt.new_lines}.values())
        # many new lines: one colour per kind and a summary legend, else one colour and entry per line
        compact = len(uniq) > 12
        for i, ln in enumerate(uniq):
            is_bus = ln.mode == "bus"
            col = ("#eb6834" if is_bus else "#d6336c") if compact else NEW_COLORS[i % len(NEW_COLORS)]
            ls = "-" if is_bus else (0, (4, 1.5))
            gx, gy = _merc(city, *ln.meta["shape"]) if "shape" in ln.meta else (ox[ln.stops], oy[ln.stops])
            ax.plot(gx, gy, color=col, lw=(2 if compact else 3) if is_bus else 4, ls=ls, zorder=8,
                    alpha=0.85 if compact else 1.0)
            ax.scatter(ox[ln.stops], oy[ln.stops], s=9 if is_bus else 22, color=col,
                       edgecolor="white", lw=0.8, zorder=9)
            if compact:
                continue
            kind = "otobüs" if is_bus else "metro"
            ax.annotate(f"{ln.name}", (ox[ln.stops[0]], oy[ln.stops[0]]), fontsize=8, fontweight="bold",
                        color="white", zorder=10, bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="none"))
            handles.append(Line2D([], [], color=col, lw=3, ls=ls, marker="o", ms=4, label=f"YENİ {kind}: {ln.name}"))
        if compact:
            n_bus = sum(ln.mode == "bus" for ln in uniq)
            if len(uniq) - n_bus:
                handles.append(Line2D([], [], color="#d6336c", lw=3, ls=(0, (4, 1.5)), marker="o", ms=4,
                                      label=f"YENİ metro hatları ({len(uniq) - n_bus} hat)"))
            if n_bus:
                handles.append(Line2D([], [], color="#eb6834", lw=2, marker="o", ms=3,
                                      label=f"YENİ otobüs hatları ({n_bus} hat)"))
        ext_seen = set()
        for ln in opt.lines:
            ext = ln.meta.get("ext_stops")
            if not ext or ln.name in ext_seen:
                continue
            ext_seen.add(ln.name)
            pos = [k for k, s in enumerate(ln.stops) if s in set(ext)]
            for run in _runs(pos):  # each extension incl. the old terminal it starts from
                k0, k1 = max(run[0] - 1, 0), min(run[-1] + 1, len(ln.stops) - 1)
                seg = ln.stops[k0:k1 + 1]
                ax.plot(ox[seg], oy[seg], color="#6250d6", lw=4.5, ls=(0, (4, 1.5)), zorder=8)
                ax.scatter(ox[seg], oy[seg], s=22, color="#6250d6", edgecolor="white", lw=0.8, zorder=9)
            ax.annotate(f"{_short(ln.name)}+" if compact else f"{_short(ln.name)} uzatma",
                        (ox[ln.stops[pos[-1]]], oy[ln.stops[pos[-1]]]), fontsize=7 if compact else 8,
                        fontweight="bold", color="white", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="#6250d6", ec="none"))
        if ext_seen:
            handles.append(Line2D([], [], color="#6250d6", lw=3, ls=(0, (4, 1.5)), marker="o", ms=4,
                                  label=f"raylı hat uzatması ({len(ext_seen)} hat)"))
    # frame the urban area: where 93% of the population lives, plus every new line
    zx, zy = _merc(city, z.x, z.y)
    order = np.argsort(z.pop)[::-1]
    core = order[: np.searchsorted(np.cumsum(z.pop[order]), 0.93 * z.pop.sum()) + 1]
    fx, fy = list(zx[core]), list(zy[core])
    for ln in base.lines:  # and the whole rail network
        if ln.mode in ("metro", "rail", "tram"):
            fx += list(bx[ln.stops])
            fy += list(by[ln.stops])
    if opt is not None:
        for ln in opt.lines:
            if ln.is_new or ln.meta.get("ext_stops"):
                fx += list(ox[ln.stops])
                fy += list(oy[ln.stops])
    pad = 1500
    x0, x1, y0, y1 = min(fx) - pad, max(fx) + pad, min(fy) - pad, max(fy) + pad
    # size the figure to the content so the basemap is neither stretched nor padded
    ax.set_aspect("equal")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.set_size_inches(15, min(13, max(6, 12.5 * (y1 - y0) / (x1 - x0) + 1.2)))
    try:
        import contextily as cx
        cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.Esri.WorldStreetMap, zorder=1,
                       attribution_size=6)
    except Exception as e:  # offline: draw without basemap
        ax.text(0.01, 0.01, f"basemap unavailable: {e}", transform=ax.transAxes, fontsize=6)
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, framealpha=0.92)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def web_map(city: City, base: Scenario, opt: Scenario | None, path: Path, title: str = "") -> None:
    """Interactive Leaflet map (zoom, layer toggles, stop pop-ups)."""
    import folium
    import branca.colormap as bcm

    z = city.zones
    lon, lat = city.proj.to_lonlat(z.x, z.y)
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles © Esri", name="Esri World Street Map").add_to(m)

    dens = z.pop / (z.cell_size / 1000) ** 2
    cmap = bcm.LinearColormap(["#ffffcc", "#fd8d3c", "#bd0026"], vmin=0, vmax=float(np.percentile(dens, 98)),
                              caption="Nüfus yoğunluğu (kişi/km²)")
    g_pop = folium.FeatureGroup(name="Nüfus (WorldPop 2026)", show=True)
    h = z.cell_size / 2
    for i in range(len(z)):
        lo, la = city.proj.to_lonlat([z.x[i] - h, z.x[i] + h], [z.y[i] - h, z.y[i] + h])
        folium.Rectangle([[float(la[0]), float(lo[0])], [float(la[1]), float(lo[1])]], weight=0, fill=True,
                         fill_color=cmap(min(dens[i], cmap.vmax)), fill_opacity=0.4,
                         tooltip=f"nüfus {z.pop[i]:,.0f} · iş endeksi {z.jobs[i]:,.0f}").add_to(g_pop)
    g_pop.add_to(m)
    cmap.add_to(m)

    slon, slat = city.proj.to_lonlat(base.stops["x"].to_numpy(), base.stops["y"].to_numpy())
    g_rail = folium.FeatureGroup(name="Mevcut raylı sistem + Metrobüs", show=True)
    g_bus = folium.FeatureGroup(name="Mevcut otobüs hatları", show=False)
    g_ferry = folium.FeatureGroup(name="Vapur", show=False)
    seen = set()
    for ln in base.lines:
        if ln.name in seen:
            continue
        seen.add(ln.name)
        coords = [[float(slat[i]), float(slon[i])] for i in ln.stops]
        if ln.mode == "bus":
            folium.PolyLine(coords, color="#6b7a8f", weight=1.5, opacity=0.5, tooltip=_short(ln.name)).add_to(g_bus)
        elif ln.mode == "ferry":
            folium.PolyLine(coords, color="#3a9bd9", weight=2, dash_array="6", tooltip=_short(ln.name)).add_to(g_ferry)
        else:
            col = "#c0392b" if ln.mode == "brt" else "#1f4e9c"
            folium.PolyLine(coords, color=col, weight=5, opacity=0.85, tooltip=_short(ln.name)).add_to(g_rail)
    for g in (g_bus, g_ferry, g_rail):
        g.add_to(m)

    if opt is not None:
        olon, olat = city.proj.to_lonlat(opt.stops["x"].to_numpy(), opt.stops["y"].to_numpy())
        names = opt.stops["name"].to_numpy()
        seen_new = []
        for ln in opt.new_lines:
            if ln.name in seen_new:
                continue
            seen_new.append(ln.name)
            col = NEW_COLORS[(len(seen_new) - 1) % len(NEW_COLORS)]
            kind = "metro" if ln.mode == "metro" else "otobüs"
            hw = np.nanmin(ln.headway_min)
            g = folium.FeatureGroup(name=f"YENİ {kind}: {ln.name}", show=True)
            if "shape" in ln.meta:
                glo, gla = city.proj.to_lonlat(*ln.meta["shape"])
                coords = [[float(b), float(a)] for a, b in zip(glo, gla)]
            else:
                coords = [[float(olat[i]), float(olon[i])] for i in ln.stops]
            folium.PolyLine(coords, color=col, weight=6, opacity=0.95,
                            dash_array=None if ln.mode == "bus" else "10 4",
                            tooltip=f"YENİ {kind} {ln.name} · {len(ln.stops)} durak · zirve sıklığı {hw:.0f} dk").add_to(g)
            for k, i in enumerate(ln.stops):
                existing = not str(opt.stops["stop_id"].iat[i]).startswith(ln.name)
                label = f"{ln.name} durak {k + 1}" + (f" (mevcut durak: {names[i]})" if existing else " (yeni durak)")
                folium.CircleMarker([float(olat[i]), float(olon[i])], radius=4 if ln.mode == "bus" else 6,
                                    color="white", weight=1.5, fill=True, fill_color=col, fill_opacity=1,
                                    tooltip=label).add_to(g)
            g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:6px 10px;'
            f'border-radius:6px;font:600 14px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">{title}</div>'))
    m.save(str(path))


def _runs(pos: list[int]) -> list[list[int]]:
    """Split sorted positions into runs of consecutive integers."""
    runs: list[list[int]] = []
    for p in pos:
        if runs and p == runs[-1][-1] + 1:
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs


def plot_frontier(steps: list[dict], base: dict, levels: list[dict], path: Path, target: float = 50.0) -> None:
    """Passenger-weighted daily mean door-to-door time (line) and the spread across the
    evaluated hours (band, fastest to slowest hour) against new rail km, from the full
    evaluations of today's network and each saved checkpoint. Extra buses are labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    pts = [(0.0, 0, base)] + sorted(((lv["rail_km"], lv["buses"], lv["evaluation"]) for lv in levels),
                                    key=lambda p: (p[0], p[1]))
    x = np.array([p[0] for p in pts])
    daily = np.array([p[2]["daily"]["mean_t"] for p in pts])
    lo = np.array([min(h["mean_t"] for h in p[2]["hours"].values()) for p in pts])
    hi = np.array([max(h["mean_t"] for h in p[2]["hours"].values()) for p in pts])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.fill_between(x, lo, hi, color=blue, alpha=0.14, lw=0, label="en hızlı – en yavaş saat (08–20)")
    ax.plot(x, daily, color=blue, lw=2, marker="o", ms=8, mec="white", mew=2, label="günlük ortalama (yolcu ağırlıklı)")
    for k, (km, buses, ev) in enumerate(pts):
        label = f"{daily[k]:.1f}"
        if buses:
            label += f"\n+{buses} otb."
        same_x = [q for q in range(len(pts)) if abs(pts[q][0] - km) < 25]
        right = len(same_x) > 1 and k == same_x[-1]  # stacked points: put the lower label to the right
        ax.annotate(label, (km, daily[k]), xytext=(10, -2) if right else (0, 10), textcoords="offset points",
                    ha="left" if right else "center", va="center" if right else "bottom", fontsize=8, color=ink)
    ax.axhline(target, color=muted, lw=1.2, ls=(0, (5, 3)))
    ax.annotate(f"hedef {target:.0f} dk", (0, target), xytext=(4, -12), textcoords="offset points",
                fontsize=9, color=muted)
    ax.set_xlabel("yeni raylı sistem (km; bugünkü ağ ≈ 378 km)", color=muted)
    ax.set_ylabel("kapıdan kapıya ortalama süre (dk)", color=muted)
    ax.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for sname in ("top", "right"):
        ax.spines[sname].set_visible(False)
    for sname in ("left", "bottom"):
        ax.spines[sname].set_color(grid)
    ax.tick_params(colors=muted)
    ax.set_ylim(min(target, lo.min()) - 2, hi.max() + 2)
    ax.set_xlim(-30, x.max() * 1.12 + 10)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=ink)
    ax.set_title("Yatırım arttıkça kapıdan kapıya süre (tam değerlendirme, 08:00–20:00 her 2 saatte)",
                 color=ink, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
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
