"""
Destination: L5 Sun-synchronized quasi-periodic libration orbits
(long-period in-plane amplitudes Ax ∈ {15 000, 30 000, 45 000} km).

The L5 counterpart of destinations/l4_qpo.py — same strategy (propagate to a
capture sphere around L5, then nearest-point insertion onto any of three
precomputed quasi-periodic libration orbits).  Orbit geometry is cached to
cache/l5_qpos.npz.  Propagation / insertion / pickling live in l45_base.py, and
the "Sun-synchronized" naming is explained in physics/triangular.py.
"""

import os

from .l45_base import TriangularOrbitDestination, load_or_compute_qpos
from physics.triangular import build_l5_qpos, L5

# ── Constants ─────────────────────────────────────────────────────────────────

QPO_AX_KM = [15_000, 30_000, 45_000]

_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "cache", "l5_qpos.npz"
)


def qpo_orbits():
    """L5 quasi-periodic orbit set mapped to the destination dict shape."""
    return [{
        "id":     f"l5_ax_{q['amp_km']}",
        "label":  f"{q['amp_km']:,} km L5 libration",
        "states": q["states"],
    } for q in load_or_compute_qpos(
        _CACHE_FILE, lambda: build_l5_qpos(QPO_AX_KM), "L5"
    )]


# ── Destination class ─────────────────────────────────────────────────────────

class L5QuasiPeriodic(TriangularOrbitDestination):

    id    = "l5_qpo"
    label = "L5 Sun-synchronized Quasi-periodic Orbits"
    approach_center = L5

    def _build_orbits(self):
        return qpo_orbits()


# ── Singleton and registry ────────────────────────────────────────────────────

L5_QPO_DEST = L5QuasiPeriodic()

ALL_DESTINATIONS = {
    L5_QPO_DEST.id: L5_QPO_DEST,
}
