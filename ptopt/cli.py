"""Command line interface.

    ptopt download --config configs/istanbul_pilot.yaml
    ptopt build    --config configs/istanbul_pilot.yaml
    ptopt evaluate --config configs/istanbul_pilot.yaml --hours 8 18
    ptopt optimize --config configs/istanbul_pilot.yaml
    ptopt calibrate --config configs/istanbul.yaml   # needs GOOGLE_MAPS_API_KEY / TOMTOM_API_KEY
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from . import calibrate
from .candidates import generate_candidates
from .city import download_all, load_city
from .config import load_config
from .optimize import Optimizer, optimize_frequencies
from .report import plot_map, scenario_geojson, summary_markdown, web_map, write_json, zones_geojson
from .simulate import Evaluator, corridor_times

log = logging.getLogger("ptopt")


def _calibration_path(cfg) -> Path:
    return Path(cfg["data_dir"]) / "processed" / f"calibration_{cfg['name']}.json"


def _city(args, calibrated: bool = True):
    over = {}
    if getattr(args, "hours", None):
        over.setdefault("evaluation", {})["hours"] = args.hours
    if getattr(args, "K", None):
        over.setdefault("evaluation", {})["K"] = args.K
    if getattr(args, "M", None):
        over.setdefault("evaluation", {})["M"] = args.M
    cfg = load_config(args.config, over)
    t = time.time()
    city = load_city(cfg, use_cache=not getattr(args, "rebuild", False))
    if calibrated and cfg["calibration"]["use"]:
        calib = calibrate.load(_calibration_path(cfg))
        if calib:
            calibrate.apply_calibration(city, calib)
            log.info("calibration applied: %s", {k: round(v, 2) for k, v in calib["transit"]["params"].items()}
                     if calib.get("transit") else "car only")
    modes = collections.Counter(ln.mode for ln in city.base.lines)
    log.info("city %s: %d zones (pop %.0f), %d stops, %d line patterns %s [%.1fs]", cfg["name"], len(city.zones),
             city.zones.pop.sum(), len(city.base.stops), len(city.base.lines), dict(modes), time.time() - t)
    return city


def cmd_calibrate(args):
    """Fit car speeds and transit parameters to Google / TomTom travel times."""
    city = _city(args, calibrated=False)
    cc = city.cfg["calibration"]
    hours = city.cfg["evaluation"]["hours"]
    s = calibrate.sample_pairs(city, cc["n_per_hour"], hours)
    log.info("calibration sample: %d trips", len(s.o))
    raw = calibrate.collect(city, s, Path(city.cfg["data_dir"]) / "raw" / "external" / f"cache_{city.cfg['name']}.json",
                            max_google=cc["max_google_elements"], max_tomtom=cc["max_tomtom_requests"])
    for k, v in raw.items():
        log.info("%s: %d of %d trips answered", k, int(np.isfinite(v).sum()), len(v))
    traffic, car = calibrate.fit_car(city, s)
    log.info("car: %s", car["stats"])
    transit = calibrate.fit_transit(city, s, traffic, max_evals=cc["max_evals"])
    log.info("transit: %s", transit)
    calib = {"car": car, "transit": transit, "n_trips": int(len(s.o)), "hours": hours}
    calibrate.save(calib, _calibration_path(city.cfg))
    out = Path(args.out or f"results/{city.cfg['name']}")
    # stats only (no raw provider answers) for the repository
    write_json({"car": car["stats"], "transit": {"params": transit["params"], "stats": transit["stats"]},
                "n_trips": calib["n_trips"], "hours": hours}, out / "calibration.json")
    print(json.dumps({"car": car["stats"], "transit": transit["stats"], "params": transit["params"]},
                     indent=2, ensure_ascii=False))


def cmd_frontier(args):
    """Greedy investment path within budget caps; full evaluation + maps at checkpoints,
    stopping once the passenger-weighted daily mean reaches the target (if set)."""
    import csv

    from .frontier import run_frontier
    from .report import plot_frontier

    city = _city(args)
    fc = city.cfg["frontier"]
    out = Path(args.out or f"results/{city.cfg['name']}") / fc.get("out_dir", "frontier")
    out.mkdir(parents=True, exist_ok=True)
    full = Evaluator(city)
    base = full.evaluate(city.base).summary()
    levels = []

    def checkpoint(level, sc, selected, steps):
        t = time.time()
        res = full.evaluate(sc).summary()
        tag = "target" if level == "target" else f"level_{int(round(level * 100)):03d}"
        d = out / tag
        d.mkdir(exist_ok=True)
        with open(d / "scenario.pkl", "wb") as f:
            pickle.dump(sc, f)
        buses, km = steps[-1]["buses"], steps[-1]["rail_km"]
        title = f"İstanbul: +{buses} otobüs, +{km:.0f} km raylı (günlük ort. {res['daily']['mean_t']:.1f} dk)"
        write_json(scenario_geojson(city, sc, only_new=True), d / "new_lines.geojson")
        plot_map(city, city.base, sc, d / "map.png", title)
        web_map(city, city.base, sc, d / "map.html", title)
        info = {"level": level, "spend": steps[-1]["spend"], "actions": [a.label() for a in selected],
                "evaluation": res, "buses": buses, "rail_km": km}
        write_json(info, d / "summary.json")
        levels.append(info)
        log.info("checkpoint %s: daily %.1f, hours %s [%.0fs]", tag, res["daily"]["mean_t"],
                 {h: round(v["mean_t"], 1) for h, v in res["hours"].items()}, time.time() - t)
        return res["daily"]["mean_t"]

    res = run_frontier(city, on_checkpoint=checkpoint)
    with open(out / "frontier.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for s in res["steps"] for k in s}, key=lambda k: (k != "step", k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(res["steps"])
    target = fc.get("target_min", 50)
    plot_frontier(res["steps"], base, levels, out / "frontier.png", target=target)
    report = {"inventory": res["inventory"], "budget": res["budget"], "base": base, "levels": levels,
              "steps": res["steps"]}
    write_json(report, out / "frontier.json")
    md = frontier_markdown(report, target)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)


def frontier_markdown(rep: dict, target: float) -> str:
    inv, bud, base = rep["inventory"], rep["budget"], rep["base"]
    hours = list(base["hours"])
    lines = [f"Bugünkü ağ: zirve otobüs filosu ≈ {inv['bus_fleet_peak']:.0f}, raylı sistem ≈ {inv['rail_km']:.0f} km "
             f"({inv['rail_lines']} hat).",
             f"Bütçe üst sınırı: +{bud['extra_buses']:.0f} otobüs (filonun %{100 * bud['extra_buses'] / max(inv['bus_fleet_peak'], 1):.0f}'i), "
             f"+{bud['rail_km']:.0f} km raylı sistem (ağın %{100 * bud['rail_km'] / max(inv['rail_km'], 1):.0f}'i).",
             f"Değerlendirilen saatler: {', '.join(f'{h}:00' for h in hours)}. Günlük ortalama, her saatin yolcu "
             f"hacmiyle ağırlıklıdır.", "",
             "| seviye | ek otobüs | yeni raylı km | günlük ort. | " + " | ".join(f"{h}:00" for h in hours) + " |",
             "|---|---|---|---|" + "---|" * len(hours)]

    def row(name, buses, km, ev):
        return (f"| {name} | {buses} | {km:.0f} | **{ev['daily']['mean_t']:.1f}** | "
                + " | ".join(f"{ev['hours'][h]['mean_t']:.1f}" for h in hours) + " |")

    lines.append(row("bugün", 0, 0.0, base))
    for lv in sorted(rep["levels"], key=lambda lv: lv["spend"]):
        name = "hedef" if lv["level"] == "target" else f"%{lv['spend'] * 100:.0f} bütçe"
        lines.append(row(name, lv["buses"], lv["rail_km"], lv["evaluation"]))
    reached = [lv for lv in rep["levels"] if lv["evaluation"]["daily"]["mean_t"] <= target]
    lines.append("")
    if reached:
        lv = min(reached, key=lambda lv: lv["spend"])
        lines.append(f"**Hedef {target:.0f} dk'ya ulaşıldı:** +{lv['buses']} otobüs (bugünkü filonun "
                     f"%{100 * lv['buses'] / max(inv['bus_fleet_peak'], 1):.0f}'i) ve +{lv['rail_km']:.0f} km raylı sistem "
                     f"(bugünkü ağın %{100 * lv['rail_km'] / max(inv['rail_km'], 1):.0f}'i).")
    elif rep["levels"]:
        best = min(lv["evaluation"]["daily"]["mean_t"] for lv in rep["levels"])
        lines.append(f"**Hedef {target:.0f} dk'ya bu bütçe üst sınırıyla ulaşılamadı** (en iyi günlük ortalama {best:.1f} dk).")
    lines += ["", "**Seçilen yatırımlar (sırayla)**", ""]
    lines += [f"{s['step']}. {s['action']} → günlük ort. {s.get('mean_t_daily', float('nan')):.1f} dk"
              for s in rep["steps"][1:]]
    return "\n".join(lines) + "\n"


def cmd_reallocate(args):
    """Fleet-neutral bus frequency reallocation (square-root rule) on a saved scenario."""
    city = _city(args)
    with open(args.scenario, "rb") as f:
        sc = pickle.load(f)
    full = Evaluator(city)
    before = full.evaluate(sc).summary()
    sc2, rep = optimize_frequencies(city, sc, log_cb=lambda r: log.info("frequencies %s", r))
    after = full.evaluate(sc2).summary()
    out = Path(args.out or Path(args.scenario).parent.with_name(Path(args.scenario).parent.name + "_realloc"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scenario.pkl", "wb") as f:
        pickle.dump(sc2, f)
    write_json({"before": before, "after": after, "periods": rep}, out / "summary.json")
    for h in before["hours"]:
        print(f"{h}:00  {before['hours'][h]['mean_t']:.1f} -> {after['hours'][h]['mean_t']:.1f} dk")


def cmd_explain(args):
    """Where the time goes: walk / wait / ride per distance band, today vs a saved scenario."""
    from .analysis import time_breakdown

    city = _city(args)
    hour = (args.hours or [8])[0]
    rows = {"bugün": time_breakdown(city, city.base, hour)}
    if args.scenario:
        with open(args.scenario, "rb") as f:
            rows["senaryo"] = time_breakdown(city, pickle.load(f), hour)
    out = Path(args.out or f"results/{city.cfg['name']}")
    write_json(rows, out / f"time_breakdown_{hour}.json")
    keys = [("mean_total", "toplam"), ("access_egress", "durağa/duraktan yürüme"), ("wait", "bekleme"),
            ("in_vehicle", "araç içi"), ("transfer_walk", "aktarma yürüyüşü"),
            ("boardings_per_transit_trip", "biniş sayısı"), ("share_unreachable", "bağlantısız pay")]
    print(f"| {hour}:00 | " + " | ".join(rows) + " |\n|---|" + "---|" * len(rows))
    for k, label in keys:
        vals = [r["all"][k] for r in rows.values()]
        fmt = (lambda v: f"{v:.1%}") if k.startswith("share") else (lambda v: f"{v:.1f}")
        print(f"| {label} | " + " | ".join(fmt(v) for v in vals) + " |")


def cmd_download(args):
    download_all(load_config(args.config))


def cmd_build(args):
    city = _city(args)
    out = Path(args.out or f"results/{city.cfg['name']}")
    write_json(zones_geojson(city), out / "zones.geojson")
    write_json(scenario_geojson(city, city.base), out / "network_base.geojson")
    plot_map(city, city.base, None, out / "map_base.png", f"{city.cfg['name']}: nüfus ve mevcut ağ")
    web_map(city, city.base, None, out / "map_base.html", f"{city.cfg['name']}: mevcut ağ")
    log.info("hourly demand profile: %s", np.round(city.hourly_profile * 100, 1).tolist())


def cmd_evaluate(args):
    city = _city(args)
    ev = Evaluator(city)
    t = time.time()
    res = ev.evaluate(city.base).summary()
    res["corridors"] = corridor_times(city, city.base, ev.hours)
    out = Path(args.out or f"results/{city.cfg['name']}")
    write_json(res, out / "evaluation_base.json")
    print(summary_markdown(res, None, res["corridors"], None, None))
    log.info("evaluation took %.1fs", time.time() - t)


def cmd_optimize(args):
    city = _city(args)
    cfg = city.cfg
    oc = cfg["optimization"]
    out = Path(args.out or f"results/{cfg['name']}")
    full = Evaluator(city)
    fast = Evaluator(city, K=oc["eval_K"], M=oc["eval_M"], n_origins=oc.get("eval_origins"))
    t = time.time()
    base = full.evaluate(city.base).summary()
    log.info("base objective %.2f [%.1fs]", base["objective"], time.time() - t)

    cands = generate_candidates(city, fast)
    log.info("%d candidate lines", len(cands))
    opt = Optimizer(city, fast, cands)
    res = opt.run()
    log.info("lines: %s  J %.2f -> %.2f  cost %s", [c.cid for c in res.selection], res.base_objective,
             res.objective, opt.cost(res.selection))

    sc = res.scenario
    freq_report = []
    if not args.skip_frequencies:
        sc, freq_report = optimize_frequencies(city, sc, log_cb=lambda r: log.info("frequencies %s", r))

    final = full.evaluate(sc).summary()
    cb = corridor_times(city, city.base, full.hours)
    co = corridor_times(city, sc, full.hours)
    sel_info = []
    for c in res.selection:
        sel_info.append({"line": c.cid, "mode": c.mode, "stops": len(c.xs), "length_km": c.length_km,
                         "peak_headway": float(np.nanmin(c.headway_min))})
    buses, km = opt.cost(res.selection)
    report = {"base": base, "optimised": final, "corridors_base": cb, "corridors_optimised": co,
              "new_lines": sel_info, "cost": {"extra_buses": buses, "metro_km": km},
              "frequency_periods": freq_report, "history": res.history,
              "runtime_s": time.time() - t}
    write_json(report, out / "optimisation.json")
    write_json(scenario_geojson(city, sc, only_new=True), out / "new_lines.geojson")
    write_json(scenario_geojson(city, sc), out / "network_optimised.geojson")
    with open(out / "scenario_optimised.pkl", "wb") as f:
        pickle.dump(sc, f)
    _maps(city, sc, out)
    md = summary_markdown(base, final, cb, co, sel_info)
    md += f"\nBudget used: {buses} extra buses, {km:.1f} metro km.\n"
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)


def _maps(city, sc, out):
    name = city.cfg["name"]
    plot_map(city, city.base, sc, out / "map_optimised.png", f"{name}: önerilen yeni hatlar")
    web_map(city, city.base, sc, out / "map_optimised.html", f"{name}: önerilen yeni hatlar")


def cmd_map(args):
    """Redraw maps from a saved optimisation result."""
    city = _city(args)
    out = Path(args.out or f"results/{city.cfg['name']}")
    with open(out / "scenario_optimised.pkl", "rb") as f:
        sc = pickle.load(f)
    _maps(city, sc, out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ptopt", description="Public transport line & frequency optimisation")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("download", cmd_download), ("build", cmd_build), ("evaluate", cmd_evaluate),
                     ("optimize", cmd_optimize), ("map", cmd_map), ("calibrate", cmd_calibrate),
                     ("frontier", cmd_frontier), ("explain", cmd_explain), ("reallocate", cmd_reallocate)):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--out")
        s.add_argument("--rebuild", action="store_true", help="ignore the processed-city cache")
        s.add_argument("--hours", type=int, nargs="+")
        s.add_argument("--K", type=int)
        s.add_argument("--M", type=int)
        if name == "optimize":
            s.add_argument("--skip-frequencies", action="store_true")
        if name in ("explain", "reallocate"):
            s.add_argument("--scenario", required=name == "reallocate",
                           help="pickled scenario, e.g. results/istanbul/frontier/level_100/scenario.pkl")
        s.set_defaults(fn=fn)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args.fn(args)


if __name__ == "__main__":
    main()
