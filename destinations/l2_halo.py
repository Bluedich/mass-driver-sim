"""
Destination: L2 Halo Orbits (northern family, Az ∈ {5 000, 10 000, 20 000, 30 000} km).

The L2 counterpart of destinations/l1_halo.py — same strategy (propagate to an
approach sphere centred on L2, then nearest-point insertion onto any of four
precomputed northern halo orbits).  Orbit geometry is cached to
cache/l2_halos.npz.  Propagation / insertion / pickling live in l1_base.py.
"""

import os
import warnings

import numpy as np

from .l1_base import L1OrbitDestination
from physics.halo import build_l2_halos, L2_X

# ── Constants ─────────────────────────────────────────────────────────────────

HALO_AZ_KM = [5_000, 10_000, 20_000, 30_000]

_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "cache", "l2_halos.npz"
)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _save_halos(path, halos):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = {
        "n":     np.array(len(halos)),
        "az_km": np.array([h["az_km"] for h in halos]),
        "T":     np.array([h["T"]     for h in halos]),
    }
    for i, h in enumerate(halos):
        kw[f"states_{i}"] = h["states"]
    np.savez(path, **kw)


def _load_halos(path):
    data = np.load(path)
    n = int(data["n"])
    return [
        {
            "az_km":  int(data["az_km"][i]),
            "T":      float(data["T"][i]),
            "states": data[f"states_{i}"],
        }
        for i in range(n)
    ]


def load_or_compute_halos():
    """L2 halo orbit set, loaded from cache/l2_halos.npz or computed and cached."""
    cache = os.path.abspath(_CACHE_FILE)

    if os.path.exists(cache):
        try:
            halos = _load_halos(cache)
            if halos:
                return halos
        except Exception as exc:
            warnings.warn(f"L2 halo cache load failed ({exc}); recomputing.")

    halos = build_l2_halos(HALO_AZ_KM)

    try:
        _save_halos(cache, halos)
    except Exception as exc:
        warnings.warn(f"Could not save L2 halo cache: {exc}")

    return halos


def halo_orbits():
    """L2 halo orbit set mapped to the L1OrbitDestination dict shape."""
    return [{
        "id":     f"l2_az_{h['az_km']}",
        "label":  f"{h['az_km']:,} km L2 halo",
        "states": h["states"],
    } for h in load_or_compute_halos()]


# ── Destination class ─────────────────────────────────────────────────────────

class L2Halo(L1OrbitDestination):

    id    = "l2_halo"
    label = "L2 Halo Orbits (5 000–30 000 km)"
    approach_center_x = L2_X

    def _build_orbits(self):
        return halo_orbits()


# ── Singleton and registry ────────────────────────────────────────────────────

L2_HALO_DEST = L2Halo()

ALL_DESTINATIONS = {
    L2_HALO_DEST.id: L2_HALO_DEST,
}
