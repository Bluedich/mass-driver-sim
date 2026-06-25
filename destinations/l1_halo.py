"""
Destination: L1 Halo Orbits (northern family, Az ∈ {5 000, 10 000, 20 000, 30 000} km).

Strategy
--------
1. Precompute four northern L1 halo orbits via differential correction
   (physics/halo.py).  Results are cached to cache/l1_halos.npz so the
   ~3-second computation only runs once.
2. Propagate from the Moon surface until the spacecraft enters a sphere of
   radius R_APPROACH centred on L1.
3. At that crossing state, find the nearest point (by position) on each
   precomputed halo orbit and compute ΔV = |v_sc − v_halo|.
4. Return the minimum ΔV across all four orbit sizes.

Pickling
--------
L1Halo.__getstate__ forces halo computation before pickling so that worker
processes in the ProcessPoolExecutor receive the precomputed numpy arrays
directly, without recomputing.
"""

import os
import threading
import warnings

import numpy as np

from .base import Destination
from physics.cr3bp import (
    MU, DU_KM, VU_KMS,
    make_moon_impact_event, make_earth_impact_event, make_escape_event,
    propagate,
)
from physics.halo import make_l1_approach_event, build_l1_halos

# ── Constants ─────────────────────────────────────────────────────────────────

HALO_AZ_KM  = [5_000, 10_000, 20_000, 30_000]

R_APPROACH  = 0.09    # DU ≈ 34 600 km — capture sphere around L1
T_MAX_TU    = 12.0    # TU ≈ 52 days — long enough for manifold transfers
MAX_STEP    = 0.02    # TU ≈ 18 min
RTOL        = 1e-8
ATOL        = 1e-10

_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "cache", "l1_halos.npz"
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


def _load_or_compute_halos():
    cache = os.path.abspath(_CACHE_FILE)

    if os.path.exists(cache):
        try:
            halos = _load_halos(cache)
            if halos:
                return halos
        except Exception as exc:
            warnings.warn(f"L1 halo cache load failed ({exc}); recomputing.")

    halos = build_l1_halos(HALO_AZ_KM)

    try:
        _save_halos(cache, halos)
    except Exception as exc:
        warnings.warn(f"Could not save L1 halo cache: {exc}")

    return halos


# ── Destination class ─────────────────────────────────────────────────────────

class L1Halo(Destination):

    id    = "l1_halo"
    label = "L1 Halo Orbits (5 000–30 000 km)"
    default_insertion_mode = "prograde"
    insertion_mode         = "prograde"

    def __init__(self):
        self._halos = None
        self._lock  = threading.Lock()

    # ── Lazy halo initialisation ──────────────────────────────────────────────

    def _ensure_halos(self):
        if self._halos is not None:
            return
        with self._lock:
            if self._halos is None:
                self._halos = _load_or_compute_halos()

    # ── Pickling support (workers must receive computed arrays) ───────────────

    def __getstate__(self):
        self._ensure_halos()       # compute before pickling
        state = self.__dict__.copy()
        del state["_lock"]         # locks are not picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ── Core computation ──────────────────────────────────────────────────────

    def compute_deltav(self, state0):
        """
        Returns (dv_nd, trajectory_dict).

        dv_nd is in DU/TU.  Returns np.inf if L1 approach sphere is not reached.
        """
        self._ensure_halos()

        approach_event = make_l1_approach_event(R_APPROACH)
        moon_event     = make_moon_impact_event()
        earth_event    = make_earth_impact_event()
        escape_event   = make_escape_event(r_max=3.5)

        sol = propagate(
            state0,
            (0.0, T_MAX_TU),
            events=[approach_event, moon_event, earth_event, escape_event],
            rtol=RTOL,
            atol=ATOL,
            max_step=MAX_STEP,
        )

        traj = {
            "t": sol.t,
            "x": sol.y[0],
            "y": sol.y[1],
            "z": sol.y[2],
            "burns": [],
        }

        # approach_event is index 0 — did the spacecraft reach L1?
        if not sol.t_events[0].size:
            return np.inf, traj

        sc = sol.y_events[0][0]   # state at L1 approach

        # ── Find minimum insertion ΔV across all halo orbit sizes ─────────────
        best_dv       = np.inf
        best_halo_idx = None

        for i, halo in enumerate(self._halos):
            xyz  = halo["states"][:, :3]
            vel  = halo["states"][:, 3:]
            dists = np.sqrt(((xyz - sc[:3])**2).sum(axis=1))
            j    = int(np.argmin(dists))
            dv   = float(np.sqrt(((sc[3:] - vel[j])**2).sum()))
            if dv < best_dv:
                best_dv       = dv
                best_halo_idx = i

        traj["burns"].append({
            "x":      float(sc[0]),
            "y":      float(sc[1]),
            "z":      float(sc[2]),
            "dv_kms": best_dv * VU_KMS,
        })

        # Attach the best-matching halo orbit for 3-D visualisation
        if best_halo_idx is not None:
            h = self._halos[best_halo_idx]
            traj["halo_x"] = h["states"][:, 0]
            traj["halo_y"] = h["states"][:, 1]
            traj["halo_z"] = h["states"][:, 2]

        return best_dv, traj


# ── Singleton and registry ────────────────────────────────────────────────────

L1_HALO_DEST = L1Halo()

ALL_DESTINATIONS = {
    L1_HALO_DEST.id: L1_HALO_DEST,
}
