"""
Shared machinery for collinear libration-point orbit destinations
(L1/L2 halo, Lyapunov, …).

A subclass supplies the target orbits via _build_orbits(), returning a list of
dicts each with keys:
    'id'     : str        stable id; matched against burn['orbit_id']
    'label'  : str        shown on hover
    'states' : ndarray (N, 6)   one full period, uniformly sampled (non-dim)

This base handles:
  * lazy, thread-safe orbit construction with pickling support (so worker
    processes receive precomputed numpy arrays, never recompute);
  * propagation to the L1 approach sphere and nearest-point insertion ΔV with
    a configurable insertion direction (prograde / retrograde / both);
  * target_orbits() rendering.

Insertion direction
-------------------
At the nearest point on a target orbit the spacecraft can be inserted to
circulate the orbit in either sense.  "prograde" matches the stored orbit
velocity v; "retrograde" matches −v (the reverse traversal); "both" takes the
cheaper of the two.
"""

import threading

import numpy as np

from .base import Destination
from physics.cr3bp import (
    VU_KMS,
    make_moon_impact_event, make_earth_impact_event, make_escape_event,
    propagate,
)
from physics.halo import make_approach_event, L1_X

# ── Propagation constants (shared with the original halo destination) ─────────
R_APPROACH = 0.09    # DU ≈ 34 600 km — capture sphere around L1
T_MAX_TU   = 12.0    # TU ≈ 52 days
MAX_STEP   = 0.02    # TU ≈ 18 min
RTOL       = 1e-8
ATOL       = 1e-10


def _best_insertion(sc, orbits, mode):
    """
    Cheapest insertion of crossing state `sc` onto any orbit in `orbits`.

    Returns (best_dv_nd, best_orbit_id).
    """
    best_dv = np.inf
    best_id = None

    for o in orbits:
        states = o["states"]
        xyz    = states[:, :3]
        vel    = states[:, 3:]
        dists  = np.sqrt(((xyz - sc[:3])**2).sum(axis=1))
        j      = int(np.argmin(dists))
        v_orb  = vel[j]

        dv_pro = float(np.sqrt(((sc[3:] - v_orb)**2).sum()))
        if mode == "prograde":
            dv = dv_pro
        elif mode == "retrograde":
            dv = float(np.sqrt(((sc[3:] + v_orb)**2).sum()))
        else:   # "both"
            dv_ret = float(np.sqrt(((sc[3:] + v_orb)**2).sum()))
            dv = min(dv_pro, dv_ret)

        if dv < best_dv:
            best_dv = dv
            best_id = o["id"]

    return best_dv, best_id


class L1OrbitDestination(Destination):
    """Abstract base — subclasses implement _build_orbits()."""

    default_insertion_mode = "both"
    insertion_mode         = "both"

    # x-coordinate (DU) of the approach-sphere centre; subclasses override for
    # other libration points (e.g. L2).
    approach_center_x = L1_X

    def __init__(self):
        self._orbits = None
        self._lock   = threading.Lock()

    # ── Subclass hook ─────────────────────────────────────────────────────────

    def _build_orbits(self):
        """Return list of {'id', 'label', 'states' (N,6)}.  Override."""
        raise NotImplementedError

    # ── Lazy, thread-safe construction ────────────────────────────────────────

    def _ensure_orbits(self):
        if self._orbits is None:
            with self._lock:
                if self._orbits is None:
                    self._orbits = self._build_orbits()
        return self._orbits

    # ── Pickling support (workers must receive computed arrays) ───────────────

    def __getstate__(self):
        self._ensure_orbits()      # compute before pickling
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ── Core computation ──────────────────────────────────────────────────────

    def compute_deltav(self, state0):
        """
        Returns (dv_nd, trajectory_dict).

        dv_nd is in DU/TU.  Returns np.inf if the L1 approach sphere is not
        reached.
        """
        orbits = self._ensure_orbits()

        approach_event = make_approach_event(self.approach_center_x, R_APPROACH)
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

        best_dv, best_id = _best_insertion(sc, orbits, self.insertion_mode)

        burn = {
            "x":      float(sc[0]),
            "y":      float(sc[1]),
            "z":      float(sc[2]),
            "dv_kms": best_dv * VU_KMS,
        }
        if best_id is not None:
            burn["orbit_id"] = best_id
        traj["burns"].append(burn)

        return best_dv, traj

    # ── Visualisation ─────────────────────────────────────────────────────────

    def target_orbits(self):
        return [{
            "id":    o["id"],
            "label": o["label"],
            "x":     o["states"][:, 0],
            "y":     o["states"][:, 1],
            "z":     o["states"][:, 2],
        } for o in self._ensure_orbits()]
