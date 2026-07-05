"""
Destination: L2 Planar Lyapunov Orbits (Ax ∈ {5 000, 10 000, 15 000, 20 000} km).

The L2 counterpart of destinations/l1_lyapunov.py.  The L2 Lyapunov family stays
clear of the Moon up to ~20 000 km, so the amplitudes stop there (smaller than
the L1 set).  Orbit geometry is cached to cache/l2_lyapunovs.npz.
"""

import os
import warnings

import numpy as np

from .l1_base import L1OrbitDestination
from physics.halo import L2_X
from physics.lyapunov import build_l2_lyapunovs

# ── Constants ─────────────────────────────────────────────────────────────────

LYAP_AX_KM = [5_000, 10_000, 15_000, 20_000]

_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "cache", "l2_lyapunovs.npz"
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
    """L2 Lyapunov orbit set, loaded from cache or computed and cached."""
    cache = os.path.abspath(_CACHE_FILE)

    if os.path.exists(cache):
        try:
            orbits = _load_lyapunovs(cache)
            if orbits:
                return orbits
        except Exception as exc:
            warnings.warn(f"L2 Lyapunov cache load failed ({exc}); recomputing.")

    orbits = build_l2_lyapunovs(LYAP_AX_KM)

    try:
        _save_lyapunovs(cache, orbits)
    except Exception as exc:
        warnings.warn(f"Could not save L2 Lyapunov cache: {exc}")

    return orbits


def lyapunov_orbits():
    """L2 Lyapunov orbit set mapped to the L1OrbitDestination dict shape."""
    return [{
        "id":     f"l2_ax_{o['ax_km']}",
        "label":  f"{o['ax_km']:,} km L2 Lyapunov",
        "states": o["states"],
    } for o in load_or_compute_lyapunovs()]


# ── Destination class ─────────────────────────────────────────────────────────

class L2Lyapunov(L1OrbitDestination):

    id    = "l2_lyapunov"
    label = "L2 Lyapunov Orbits (5 000–20 000 km)"
    approach_center_x = L2_X

    def _build_orbits(self):
        return lyapunov_orbits()


# ── Singleton and registry ────────────────────────────────────────────────────

L2_LYAPUNOV_DEST = L2Lyapunov()

ALL_DESTINATIONS = {
    L2_LYAPUNOV_DEST.id: L2_LYAPUNOV_DEST,
}
