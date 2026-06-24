"""
Destination: 1 200 km circular Earth LEO in the Moon's orbit plane (i = 0°).

Strategy
--------
1. Propagate CR3BP from launch state for up to T_MAX TU.
2. Detect when the trajectory crosses r = R_TARGET from ABOVE (inbound).
   This fires whether the periapsis is exactly at the target, above it, or below.
3. At that crossing, compute ΔV = |v_inertial − v_target_orbit|
   where v_target is the prograde or retrograde circular orbit velocity in
   the equatorial plane.  Both options are tried; the cheaper burn wins.
   Out-of-plane velocity (vz) is correctly included in the DV cost.
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
    default_insertion_mode = "prograde"
    insertion_mode         = "prograde"

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
        dv_nd, dv_kms = _circularization_deltav(state_cross, mode=self.insertion_mode)

        traj["burns"].append({
            "x":      state_cross[0],
            "y":      state_cross[1],
            "z":      state_cross[2],
            "dv_kms": dv_kms,
        })

        return dv_nd, traj


def _circularization_deltav(state, mode="prograde"):
    """
    ΔV to circularise into the target orbit (0° inclination, Moon-orbit plane).

    mode : "prograde" | "retrograde" | "both"
        Controls which circular orbit direction(s) are evaluated.
        "both" takes the cheaper of the two.
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

    # In-plane radius (needed for azimuthal direction of target orbit)
    rp = np.sqrt(xe**2 + ye**2)
    if rp < 1e-10:   # polar crossing — azimuth undefined
        return np.inf, np.inf

    # Rotating → Earth-centred inertial: add Ω×r (Ω=1 in non-dim units)
    vxi = vx - ye
    vyi = vy + xe
    vzi = vz

    # Circular speed at current radius
    v_circ = np.sqrt(MU_EARTH_ND / r)

    # Target orbit velocity vectors in the equatorial plane:
    #   prograde  φ̂ = (−ye/rp,  xe/rp, 0)
    #   retrograde    = ( ye/rp, −xe/rp, 0)
    dv_pro = np.sqrt((vxi + v_circ * ye / rp)**2 +
                     (vyi - v_circ * xe / rp)**2 +
                     vzi**2)
    if mode == "prograde":
        dv_nd = dv_pro
    elif mode == "retrograde":
        dv_ret = np.sqrt((vxi - v_circ * ye / rp)**2 +
                         (vyi + v_circ * xe / rp)**2 +
                         vzi**2)
        dv_nd = dv_ret
    else:
        dv_ret = np.sqrt((vxi - v_circ * ye / rp)**2 +
                         (vyi + v_circ * xe / rp)**2 +
                         vzi**2)
        dv_nd = min(dv_pro, dv_ret)
    dv_kms = dv_nd * VU_KMS

    return dv_nd, dv_kms


# Singleton and registry
EARTH_LEO_1200 = EarthLEO1200()

ALL_DESTINATIONS = {
    EARTH_LEO_1200.id: EARTH_LEO_1200,
}
