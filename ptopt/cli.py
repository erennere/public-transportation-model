"""Command line interface.

    ptopt download --config configs/istanbul_pilot.yaml
    ptopt build    --config configs/istanbul_pilot.yaml
    ptopt evaluate --config configs/istanbul_pilot.yaml --hours 8 18
    ptopt optimize --config configs/istanbul_pilot.yaml
"""
from __future__ import annotations

import argparse
import collections
import logging
import time
from pathlib import Path

import numpy as np

from .candidates import generate_candidates
from .city import download_all, load_city
from .config import load_config
from .optimize import Optimizer, optimize_frequencies
from .report import plot_map, scenario_geojson, summary_markdown, write_json, zones_geojson
from .simulate import Evaluator, corridor_times

log = logging.getLogger("ptopt")


def _city(args):
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
    modes = collections.Counter(ln.mode for ln in city.base.lines)
    log.info("city %s: %d zones (pop %.0f), %d stops, %d line patterns %s [%.1fs]", cfg["name"], len(city.zones),
             city.zones.pop.sum(), len(city.base.stops), len(city.base.lines), dict(modes), time.time() - t)
    return city


def cmd_download(args):
    download_all(load_config(args.config))


def cmd_build(args):
    city = _city(args)
    out = Path(args.out or f"results/{city.cfg['name']}")
    write_json(zones_geojson(city), out / "zones.geojson")
    write_json(scenario_geojson(city, city.base), out / "network_base.geojson")
    plot_map(city, city.base, None, out / "map_base.png", f"{city.cfg['name']}: population and network")
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
    fast = Evaluator(city, K=oc["eval_K"], M=oc["eval_M"])
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
    plot_map(city, city.base, sc, out / "map_optimised.png", f"{cfg['name']}: proposed new lines")
    md = summary_markdown(base, final, cb, co, sel_info)
    md += f"\nBudget used: {buses} extra buses, {km:.1f} metro km.\n"
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ptopt", description="Public transport line & frequency optimisation")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("download", cmd_download), ("build", cmd_build), ("evaluate", cmd_evaluate),
                     ("optimize", cmd_optimize)):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--out")
        s.add_argument("--rebuild", action="store_true", help="ignore the processed-city cache")
        s.add_argument("--hours", type=int, nargs="+")
        s.add_argument("--K", type=int)
        s.add_argument("--M", type=int)
        if name == "optimize":
            s.add_argument("--skip-frequencies", action="store_true")
        s.set_defaults(fn=fn)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args.fn(args)


if __name__ == "__main__":
    main()
