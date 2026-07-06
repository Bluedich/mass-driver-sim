"""
Destination: L4 Sun-synchronized quasi-periodic libration orbits
(long-period in-plane amplitudes Ax ∈ {15 000, 30 000, 45 000} km).

Three quasi-periodic orbits about the triangular L4 point are precomputed
(physics/triangular.py) and cached to cache/l4_qpos.npz.  A launched trajectory
is propagated until it enters the capture sphere around L4, then the insertion
ΔV is the nearest-point velocity match onto any of the orbits (prograde /
retrograde / both — see TriangularOrbitDestination).

"Sun-synchronized" names the intended real-world station-keeping regime: these
orbits ride the long-period libration mode about L4, the mode the Sun drives in
the full Earth–Moon–Sun system.  This CR3BP model omits the Sun, so the orbits
are represented purely geometrically — see physics/triangular.py.
"""

import os

from .l45_base import TriangularOrbitDestination, load_or_compute_qpos
from physics.triangular import build_l4_qpos, L4

# ── Constants ─────────────────────────────────────────────────────────────────

QPO_AX_KM = [15_000, 30_000, 45_000]

_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "cache", "l4_qpos.npz"
)


def qpo_orbits():
    """L4 quasi-periodic orbit set mapped to the destination dict shape."""
    return [{
        "id":     f"l4_ax_{q['amp_km']}",
        "label":  f"{q['amp_km']:,} km L4 libration",
        "states": q["states"],
    } for q in load_or_compute_qpos(
        _CACHE_FILE, lambda: build_l4_qpos(QPO_AX_KM), "L4"
    )]


# ── Destination class ─────────────────────────────────────────────────────────

class L4QuasiPeriodic(TriangularOrbitDestination):

    id    = "l4_qpo"
    label = "L4 Sun-synchronized Quasi-periodic Orbits"
    approach_center = L4

    def _build_orbits(self):
        return qpo_orbits()


# ── Singleton and registry ────────────────────────────────────────────────────

L4_QPO_DEST = L4QuasiPeriodic()

ALL_DESTINATIONS = {
    L4_QPO_DEST.id: L4_QPO_DEST,
}
