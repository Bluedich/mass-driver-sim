"""
Destination: L1 Halo + Lyapunov Orbits (combined).

Exposes all eight orbits — the four northern halos and the four planar
Lyapunov orbits — as a single target set.  Insertion ΔV is the cheapest match
onto any of them (prograde / retrograde / both — see L1OrbitDestination), so
the suitability grid reflects whichever L1 orbit, of either family, is reachable
most cheaply from a given launch state.

Both orbit families reuse the caches of the standalone halo and Lyapunov
destinations, so nothing is recomputed.
"""

from .l1_base import L1OrbitDestination
from .l1_halo import halo_orbits
from .l1_lyapunov import lyapunov_orbits


class L1Combined(L1OrbitDestination):

    id    = "l1_combined"
    label = "L1 Halo + Lyapunov Orbits"

    def _build_orbits(self):
        # Halo ids are "az_*" and Lyapunov ids are "ax_*", so they never collide.
        return halo_orbits() + lyapunov_orbits()


# ── Singleton and registry ────────────────────────────────────────────────────

L1_COMBINED_DEST = L1Combined()

ALL_DESTINATIONS = {
    L1_COMBINED_DEST.id: L1_COMBINED_DEST,
}
