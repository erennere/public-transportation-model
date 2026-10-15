"""Hourly demand profile from smart-card ridership (İBB 'hourly public transport')."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ibb_hourly_profile(csv_paths, weekdays_only: bool = True, chunksize: int = 2_000_000) -> np.ndarray:
    """Share of weekday boardings per hour of day (sums to 1)."""
    counts = np.zeros(24)
    for path in csv_paths:
        for ch in pd.read_csv(path, usecols=["transition_date", "transition_hour", "number_of_passenger"],
                              chunksize=chunksize):
            if weekdays_only:
                d = pd.to_datetime(ch["transition_date"], errors="coerce")
                ch = ch[d.dt.dayofweek < 5]
            g = ch.groupby("transition_hour")["number_of_passenger"].sum()
            for h, v in g.items():
                if 0 <= int(h) < 24:
                    counts[int(h)] += v
    if counts.sum() == 0:
        raise ValueError("no ridership rows read")
    return counts / counts.sum()
