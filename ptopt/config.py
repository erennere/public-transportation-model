"""City configuration: YAML file deep-merged over defaults."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "name": "city",
    # [min_lon, min_lat, max_lon, max_lat]
    "bbox": None,
    "crs": None,  # metric CRS; UTM zone picked automatically when null
    "data_dir": "data",
    "zones": {
        "cell_size_m": 1000,
        "min_population": 50,
        "min_jobs": 20,
    },
    "population": {
        "source": "worldpop",
        "iso3": None,
        "year": 2026,
        "url": None,  # overrides the WorldPop URL template
    },
    "jobs": {
        "source": "osm",
        "overpass_url": "https://overpass.kumi.systems/api/interpreter",
        # weight of each POI tag value class (jobs proxy)
        "weights": {"office": 10.0, "shop": 3.0, "amenity": 2.0, "craft": 3.0, "industrial": 15.0},
        # share of zone population assumed to also be local jobs (services)
        "population_share": 0.1,
        "url": None,
    },
    # drivable street network for routing new bus lines ('pbf' reads jobs.pbf_url)
    "roads": {"source": "none"},
    # optional administrative boundary the study area is clipped to, e.g.
    # {name: "İstanbul", admin_level: 4} (read from jobs.pbf_url)
    "boundary": None,
    "gtfs": [],  # list of {name, path|url|files, fixes: [istanbul]}
    "traffic": {
        "source": "profile",  # 'ibb' | 'profile'
        "urls": [],
        # car free-flow speed (km/h) and hourly congestion multiplier
        "free_flow_kmh": 50.0,
        "hourly_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.8, 0.6, 0.5, 0.65, 0.75, 0.75,
                          0.75, 0.75, 0.72, 0.68, 0.6, 0.5, 0.48, 0.6, 0.75, 0.85, 0.95, 1.0],
        "weekdays_only": True,
    },
    "ridership": {"source": "profile", "urls": []},
    "demand": {
        "daily_trips_per_person": 1.6,
        "transit_capture": 1.0,  # all motorised trips are sampled; mode split is computed later
        "gravity_d0_km": 8.0,
        "furness_iterations": 30,
        # share of trips leaving per hour (normalised internally)
        "hourly_profile": [0.4, 0.2, 0.1, 0.1, 0.3, 1.2, 4.5, 8.5, 8.0, 5.0, 4.0, 4.0,
                           4.2, 4.3, 4.6, 5.5, 7.0, 8.6, 8.2, 5.8, 4.0, 3.0, 2.0, 1.0],
        # purpose shares per hour: home->work, work->home; rest is 'other'
        "share_home_work": [0.1, 0.1, 0.1, 0.2, 0.5, 0.7, 0.8, 0.8, 0.7, 0.5, 0.3, 0.2,
                            0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "share_work_home": [0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.05, 0.1, 0.15,
                            0.2, 0.25, 0.35, 0.5, 0.65, 0.7, 0.7, 0.55, 0.4, 0.3, 0.2, 0.15],
    },
    "network": {
        "walk_kmh": 4.5,
        "walk_detour": 1.3,
        "max_access_m": 800,
        "max_transfer_m": 250,
        "max_walk_only_m": 3000,
        "boarding_penalty_min": 2.0,
        "common_lines": True,  # wait = 1 / (2 x combined frequency of lines to the same next stops)
        "max_wait_min": 15.0,
        "road_detour": 1.3,
        "bus_speed_factor": 0.75,  # bus speed = car speed * factor
        "bus_max_kmh": 40.0,
        "dwell_s": 20.0,
        "mode_speed_kmh": {"metro": 38.0, "tram": 20.0, "rail": 45.0, "brt": 30.0,
                           "ferry": 22.0, "funicular": 15.0, "cable": 12.0},
        "car_access_min": 5.0,
        "rail_speed_scale": 1.0,  # multiplies mode_speed_kmh for non-bus modes (fitted by calibration)
    },
    # binary logit transit vs car for travellers who have a car available;
    # travellers without a car are captive to transit/walk
    "mode_choice": {"beta_per_min": 0.08, "asc_transit": -0.5, "car_availability": 0.5},
    "evaluation": {
        "hours": [8, 18],
        "K": 2000,
        "M": 5,
        "origins_per_sim": 200,
        "Z_min": 45.0,
        "weighting": "proportional",  # 'proportional' | 'tempered'
        "alpha": 0.5,  # used when weighting == 'tempered'
        "excess_weight": 1.0,
        "seed": 42,
        "corridors": [],  # [{name, x: [lon, lat], y: [lon, lat]}]
    },
    "candidates": {
        "n_corridors": 20,
        "min_od_km": 4.0,
        "bus_stop_spacing_m": 400,
        "metro_station_spacing_m": 1200,
        "snap_existing_m": 150,
        "density_reward": 0.6,
        "default_bus_headway_min": 10.0,
        "default_metro_headway_min": 5.0,
        "bus_corridor_width_m": 1500,  # route penalty doubles at half this distance
        "max_bus_length_km": 30.0,
    },
    "optimization": {
        "budget": {"extra_buses": 120, "metro_km": 20.0},
        "greedy_rounds": 6,
        "anneal_iters": 150,
        "anneal_t0": 0.5,
        "periods": {"am_peak": [7, 8, 9], "midday": [11, 12, 13, 14], "pm_peak": [17, 18, 19],
                    "evening": [20, 21, 22]},
        "frequency_iterations": 3,
        "frequency_modes": ["bus"],
        "min_headway_min": 2.0,
        "max_headway_min": 30.0,
        "eval_K": 600,
        "eval_M": 2,
        "eval_origins": None,  # origins per simulation while optimising (None = evaluation.origins_per_sim)
    },
    # budget–time frontier (ptopt frontier): budgets relative to today's network
    "frontier": {
        "bus_share": 0.20,       # extra buses = 20% of today's peak bus fleet
        "rail_share": 0.30,      # new rail km = 30% of today's rail network length
        "cost_hour": 8,          # fleet needs are counted at this hour
        "cheap_K": 400, "cheap_origins": 80,           # screening sample
        "fast_K": 600, "fast_M": 2, "fast_origins": 150,  # shortlist sample
        "shortlist": 6,
        "regen_every": 4,        # regenerate new-line candidates every n steps
        "boost_top": 40,         # busiest bus routes considered for more buses
        "boost_factor": 0.67,    # headway multiplier of one boost (= +50% buses)
        "max_steps": 80,
        "checkpoints": [0.25, 0.5, 0.75, 1.0],  # share of budget spent -> full evaluation + maps
        "n_jobs": None,
    },
    "calibration": {
        "use": True,             # apply data/processed/calibration_<name>.json when it exists
        "n_per_hour": 300,       # sampled trips per evaluation hour
        "max_google_elements": 4000,
        "max_tomtom_requests": 1500,
        "max_evals": 50,         # Nelder-Mead evaluations for the transit parameters
    },
    "synthetic": None,
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    user = _read_with_extends(Path(path)) if path is not None else {}
    cfg = deep_merge(DEFAULTS, user)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def _read_with_extends(path: Path) -> dict:
    """Read YAML; `extends: other.yaml` (relative to this file) is merged underneath."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        data = deep_merge(_read_with_extends(path.parent / parent), data)
    return data
