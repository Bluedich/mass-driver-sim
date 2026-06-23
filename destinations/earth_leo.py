"""
Destination: 1 200 km circular Earth LEO in the Moon's orbit plane (i = 0°).

Strategy
--------
1. Propagate CR3BP from launch state for up to T_MAX TU.
2. Detect when the trajectory crosses r = R_TARGET from ABOVE (inbound).
   This fires whether the periapsis is exactly at the target, above it, or below.
3. At that crossing, compute ΔV = √[(v_t − v_circ)² + v_r²]
   where v_r = radial speed (0 if we arrive exactly at periapsis = target altitude)
         v_t = tangential speed in Earth-centred inertial frame
   The optimiser naturally finds trajectories where periapsis ≈ target (v_r → 0).
4. If the trajectory never crosses r = R_TARGET, return ΔV = ∞.
"""

import numpy as np
from .base import Destination
from physics.cr3bp import (
    propagate,
    make_moon_impact_event, make_earth_impact_event, make_escape_event,
    MU, DU_KM, VU_KMS, R_EARTH_DU,
)

# ── Target orbit ──────────────────────────────────────────────────────────────
ALT_KM      = 1_200.0
R_TARGET_KM = 6_371.0 + ALT_KM       # 7 571 km
R_TARGET_DU = R_TARGET_KM / DU_KM    # ≈ 0.019 700 DU

MU_EARTH_ND  = 1.0 - MU
V_CIRC_ND    = np.sqrt(MU_EARTH_ND / R_TARGET_DU)
V_CIRC_KMS   = V_CIRC_ND * VU_KMS    # ≈ 7.26 km/s

T_MAX_TU = 8.0    # ~35 days
MAX_STEP = 0.01   # TU (≈ 54 min) — fine enough for Earth periapsis detection
RTOL     = 1e-7
ATOL     = 1e-9


def make_target_altitude_event():
    """
    Fires when trajectory crosses r = R_TARGET from above (inbound).
    Terminal — stops integration at the first such crossing.
    """
    def event(t, state):
        x, y, z = state[:3]
        r = np.sqrt((x + MU)**2 + y**2 + z**2)
        return r - R_TARGET_DU

    event.terminal  = True
    event.direction = -1   # inbound (r decreasing through R_TARGET)
    return event


class EarthLEO1200(Destination):

    id    = "earth_leo_1200"
    label = "1 200 km LEO (Moon orbit plane)"

    def compute_deltav(self, state0):
        """
        Returns (dv_nd, trajectory_dict).
        dv_nd is in DU/TU.  Returns np.inf if target altitude not reached.
        """
        target_event = make_target_altitude_event()
        moon_impact  = make_moon_impact_event()
        earth_impact = make_earth_impact_event()
        escape_event = make_escape_event(r_max=3.5)

        sol = propagate(
            state0,
            (0.0, T_MAX_TU),
            events=[target_event, moon_impact, earth_impact, escape_event],
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

        # Target altitude event fired? (index 0)
        if not sol.t_events[0].size:
            return np.inf, traj

        state_cross = sol.y_events[0][0]
        dv_nd, dv_kms = _circularization_deltav(state_cross)

        traj["burns"].append({
            "x":      state_cross[0],
            "y":      state_cross[1],
            "z":      state_cross[2],
            "dv_kms": dv_kms,
        })

        return dv_nd, traj


def _circularization_deltav(state):
    """
    ΔV to circularise at the target altitude crossing.

    At any point on the trajectory (not necessarily periapsis):
        ΔV = √[(v_t − v_circ)² + v_r²]
    where:
        v_r = radial speed (Earth-centred inertial)
        v_t = tangential speed (Earth-centred inertial)
        v_circ = √(μ_E / r)

    The optimiser will find trajectories where v_r → 0 (periapsis ≈ target),
    minimising ΔV naturally.
    """
    x, y, z, vx, vy, vz = state

    # Earth-centred position in rotating frame
    xe = x + MU
    ye = y
    ze = z
    r  = np.sqrt(xe**2 + ye**2 + ze**2)

    # Guard: craft inside Earth
    if r <= R_EARTH_DU:
        return np.inf, np.inf

    # Rotating → Earth-centred inertial: Ω×r = (−ye, xe, 0)
    vxi = vx - ye
    vyi = vy + xe
    vzi = vz

    # Radial unit vector
    r_hat = np.array([xe, ye, ze]) / r

    # Decompose velocity into radial and tangential
    v_vec  = np.array([vxi, vyi, vzi])
    v_r    = np.dot(v_vec, r_hat)            # radial (positive = receding)
    v_t    = np.sqrt(max(0.0, np.dot(v_vec, v_vec) - v_r**2))   # tangential magnitude

    # Circular speed at this radius
    v_circ = np.sqrt(MU_EARTH_ND / r)

    # ΔV to circularise
    dv_nd  = np.sqrt((v_t - v_circ)**2 + v_r**2)
    dv_kms = dv_nd * VU_KMS

    return dv_nd, dv_kms


# Singleton and registry
EARTH_LEO_1200 = EarthLEO1200()

ALL_DESTINATIONS = {
    EARTH_LEO_1200.id: EARTH_LEO_1200,
}
