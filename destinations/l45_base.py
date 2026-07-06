"""
Shared machinery for triangular libration-point orbit destinations (L4 / L5
Sun-synchronized quasi-periodic libration orbits).

The triangular counterpart of l1_base.py.  It reuses the same nearest-point
insertion logic (_best_insertion) but differs in two ways that matter:

  * the capture sphere is centred on an **off-axis** 3-vector (L4/L5 do not lie
    on the x-axis), so it uses physics.triangular.make_point_approach_event
    rather than the collinear make_approach_event; and
  * L4/L5 sit a full 1 DU from the Moon, so the propagation horizon and capture
    radius are larger than the collinear defaults.

Subclasses supply the target orbits via _build_orbits() and set the point
(approach_center) they belong to.  Lazy construction, pickling for worker
processes, and target_orbits() rendering are inherited from here.
"""

import os
import threading
import warnings

import numpy as np

from .base import Destination
from .l1_base import _best_insertion
from physics.cr3bp import (
    VU_KMS,
    make_moon_impact_event, make_earth_impact_event, make_escape_event,
)
from physics.triangular import make_point_approach_event
from physics.cr3bp import propagate

# ── Propagation constants ─────────────────────────────────────────────────────
# L4/L5 lie 1 DU from the Moon, so a launched trajectory needs both a wider
# capture sphere and a longer horizon than the collinear (L1/L2) destinations.
R_APPROACH = 0.32    # DU ≈ 123 000 km — encloses the libration band
T_MAX_TU   = 30.0    # TU ≈ 130 days
MAX_STEP   = 0.03    # TU ≈ 27 min
RTOL       = 1e-8
ATOL       = 1e-10


class TriangularOrbitDestination(Destination):
    """Abstract base — subclasses implement _build_orbits() and set approach_center."""

    default_insertion_mode = "both"
    insertion_mode         = "both"

    # 3-vector (DU) centre of the capture sphere — the triangular point.
    approach_center = None

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

        dv_nd is in DU/TU.  Returns np.inf if the capture sphere around the
        triangular point is not reached within T_MAX_TU.
        """
        orbits = self._ensure_orbits()

        approach_event = make_point_approach_event(self.approach_center, R_APPROACH)
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

        # approach_event is index 0 — did the spacecraft reach the point?
        if not sol.t_events[0].size:
            return np.inf, traj

        sc = sol.y_events[0][0]   # state at capture-sphere crossing

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


# ── Shared npz cache for the quasi-periodic orbit sets ────────────────────────

def _save_qpos(path, qpos):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = {
        "n":      np.array(len(qpos)),
        "amp_km": np.array([q["amp_km"] for q in qpos]),
        "T":      np.array([q["T"]      for q in qpos]),
    }
    for i, q in enumerate(qpos):
        kw[f"states_{i}"] = q["states"]
    np.savez(path, **kw)


def _load_qpos(path):
    data = np.load(path)
    n = int(data["n"])
    return [
        {
            "amp_km": int(data["amp_km"][i]),
            "T":      float(data["T"][i]),
            "states": data[f"states_{i}"],
        }
        for i in range(n)
    ]


def load_or_compute_qpos(cache_file, builder, point_label):
    """
    Quasi-periodic orbit set, loaded from `cache_file` or computed via
    `builder()` (a zero-arg callable) and cached.
    """
    cache = os.path.abspath(cache_file)

    if os.path.exists(cache):
        try:
            qpos = _load_qpos(cache)
            if qpos:
                return qpos
        except Exception as exc:
            warnings.warn(f"{point_label} QPO cache load failed ({exc}); recomputing.")

    qpos = builder()

    try:
        _save_qpos(cache, qpos)
    except Exception as exc:
        warnings.warn(f"Could not save {point_label} QPO cache: {exc}")

    return qpos
