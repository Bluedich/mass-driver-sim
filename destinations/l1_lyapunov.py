"""
Destination: L1 Planar Lyapunov Orbits (Ax ∈ {5 000, 10 000, 20 000, 30 000} km).

The planar (z = 0) analogue of the L1 halo destination.  Four Lyapunov orbits
are precomputed via differential correction (physics/lyapunov.py) and cached to
cache/l1_lyapunovs.npz.  Insertion ΔV is the nearest-point velocity match onto
any of the four orbits (prograde / retrograde / both — see L1OrbitDestination).
"""

import os
import warnings

import numpy as np

from .l1_base import L1OrbitDestination
from physics.lyapunov import build_l1_lyapunovs

# ── Constants ─────────────────────────────────────────────────────────────────

LYAP_AX_KM = [5_000, 10_000, 20_000, 30_000]

_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "cache", "l1_lyapunovs.npz"
)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _save_lyapunovs(path, orbits):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = {
        "n":     np.array(len(orbits)),
        "ax_km": np.array([o["ax_km"] for o in orbits]),
        "T":     np.array([o["T"]     for o in orbits]),
    }
    for i, o in enumerate(orbits):
        kw[f"states_{i}"] = o["states"]
    np.savez(path, **kw)


def _load_lyapunovs(path):
    data = np.load(path)
    n = int(data["n"])
    return [
        {
            "ax_km":  int(data["ax_km"][i]),
            "T":      float(data["T"][i]),
            "states": data[f"states_{i}"],
        }
        for i in range(n)
    ]


def load_or_compute_lyapunovs():
    """Lyapunov orbit set, loaded from cache or computed and cached."""
    cache = os.path.abspath(_CACHE_FILE)

    if os.path.exists(cache):
        try:
            orbits = _load_lyapunovs(cache)
            if orbits:
                return orbits
        except Exception as exc:
            warnings.warn(f"L1 Lyapunov cache load failed ({exc}); recomputing.")

    orbits = build_l1_lyapunovs(LYAP_AX_KM)

    try:
        _save_lyapunovs(cache, orbits)
    except Exception as exc:
        warnings.warn(f"Could not save L1 Lyapunov cache: {exc}")

    return orbits


def lyapunov_orbits():
    """Lyapunov orbit set mapped to the L1OrbitDestination dict shape."""
    return [{
        "id":     f"ax_{o['ax_km']}",
        "label":  f"{o['ax_km']:,} km Lyapunov",
        "states": o["states"],
    } for o in load_or_compute_lyapunovs()]


# ── Destination class ─────────────────────────────────────────────────────────

class L1Lyapunov(L1OrbitDestination):

    id    = "l1_lyapunov"
    label = "L1 Lyapunov Orbits (5 000–30 000 km)"

    def _build_orbits(self):
        return lyapunov_orbits()


# ── Singleton and registry ────────────────────────────────────────────────────

L1_LYAPUNOV_DEST = L1Lyapunov()

ALL_DESTINATIONS = {
    L1_LYAPUNOV_DEST.id: L1_LYAPUNOV_DEST,
}
