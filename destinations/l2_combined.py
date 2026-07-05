"""
Destination: L2 Halo + Lyapunov Orbits (combined).

The L2 counterpart of destinations/l1_combined.py — exposes the four L2 halos
and the four L2 Lyapunov orbits as a single target set, with insertion ΔV being
the cheapest match onto any of them.  Both families reuse the standalone L2
caches, so nothing is recomputed.
"""

from .l1_base import L1OrbitDestination
from physics.halo import L2_X
from .l2_halo import halo_orbits
from .l2_lyapunov import lyapunov_orbits


class L2Combined(L1OrbitDestination):

    id    = "l2_combined"
    label = "L2 Halo + Lyapunov Orbits"
    approach_center_x = L2_X

    def _build_orbits(self):
        # Ids are "l2_az_*" / "l2_ax_*", so the two families never collide.
        return halo_orbits() + lyapunov_orbits()


# ── Singleton and registry ────────────────────────────────────────────────────

L2_COMBINED_DEST = L2Combined()

ALL_DESTINATIONS = {
    L2_COMBINED_DEST.id: L2_COMBINED_DEST,
}
