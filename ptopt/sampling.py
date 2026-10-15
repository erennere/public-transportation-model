"""Draw K origin-destination pairs (X, Y) that follow the hour's travel flows.

Proportional weighting (default) samples pair (i, j) with probability
T_ij(h) / sum(T). If at 08:00 most people travel X->Y, most samples are X->Y,
then fewer Y->Z, fewer still Z->X, exactly in proportion to the flows; the
plain sample mean of travel time is then an unbiased estimate of the
passenger-weighted mean.

Tempered weighting samples with T_ij^alpha (alpha=0.5 is the geometric-mean
style compromise): the dominant direction still gets the majority but weak
flows are better covered. Means are then importance-reweighted with
w = T / T^alpha so they stay passenger-weighted.

Sampling is two-stage: a limited set of origins is drawn from the origin
marginal, then each sample picks one of these origins and a destination
from the conditional distribution. Cost then scales with the number of
distinct origins (one shortest-path tree each), not with K.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ODSample:
    o: np.ndarray        # origin zone per sample
    d: np.ndarray        # destination zone per sample
    weight: np.ndarray   # importance weight (normalised to mean 1)
    hour: int

    @property
    def origins(self) -> np.ndarray:
        return np.unique(self.o)


def sample_od(T: np.ndarray, K: int, n_origins: int, rng: np.random.Generator,
              weighting: str = "proportional", alpha: float = 0.5, hour: int = 0) -> ODSample:
    T = np.array(T, float)
    np.fill_diagonal(T, 0.0)
    if weighting == "proportional":
        W = T
    elif weighting == "tempered":
        W = np.power(T, alpha)
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    row = W.sum(axis=1)
    p_o = row / row.sum()
    origin_pool = rng.choice(len(p_o), size=n_origins, replace=True, p=p_o)
    o = origin_pool[rng.integers(0, n_origins, size=K)]
    d = np.empty(K, dtype=int)
    for i in np.unique(o):
        idx = np.flatnonzero(o == i)
        p = W[i] / row[i]
        d[idx] = rng.choice(len(p), size=len(idx), replace=True, p=p)
    if weighting == "proportional":
        w = np.ones(K)
    else:
        w = T[o, d] / np.maximum(W[o, d], 1e-300)
        w = w / w.mean()
    return ODSample(o=o, d=d, weight=w, hour=hour)
