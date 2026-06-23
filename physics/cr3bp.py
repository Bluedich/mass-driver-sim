"""
CR3BP (Circular Restricted 3-Body Problem) — Earth–Moon system.

Non-dimensional units
---------------------
DU  = 384 400 km          (Earth–Moon distance)
TU  = 375 190 s           (~4.343 days, 1/mean-motion)
VU  = DU/TU = 1.02454 km/s

Rotating frame
--------------
Earth at (-mu, 0, 0), Moon at (1-mu, 0, 0).
Positive x points from Earth toward Moon.
z is the orbit-normal (northward).
"""

import numpy as np
from scipy.integrate import solve_ivp

# ── Physical constants (non-dimensional) ──────────────────────────────────────
MU = 0.012150584269940356   # M_moon / (M_earth + M_moon)

# Dimensional conversion factors
DU_KM  = 384_400.0          # km per DU
TU_S   = 375_190.0          # seconds per TU
VU_KMS = DU_KM / TU_S       # km/s per VU  ≈ 1.02454

R_MOON_DU = 1_737.4 / DU_KM   # Moon radius in DU
R_EARTH_DU = 6_371.0 / DU_KM  # Earth radius in DU


def eom(t, state):
    """CR3BP equations of motion (non-dimensional)."""
    x, y, z, vx, vy, vz = state

    r1_sq = (x + MU)**2 + y**2 + z**2       # Earth-to-craft squared
    r2_sq = (x - 1 + MU)**2 + y**2 + z**2   # Moon-to-craft squared
    r1 = np.sqrt(r1_sq)
    r2 = np.sqrt(r2_sq)

    c1 = (1 - MU) / r1**3
    c2 = MU / r2**3

    ax = 2*vy + x - c1*(x + MU) - c2*(x - 1 + MU)
    ay = -2*vx + y - c1*y - c2*y
    az = -c1*z - c2*z

    return [vx, vy, vz, ax, ay, az]


def jacobi(state):
    """Jacobi constant (conserved quantity; useful for sanity checks)."""
    x, y, z, vx, vy, vz = state
    v2 = vx**2 + vy**2 + vz**2
    r1 = np.sqrt((x + MU)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + MU)**2 + y**2 + z**2)
    omega = 0.5*(x**2 + y**2) + (1 - MU)/r1 + MU/r2
    return 2*omega - v2


def propagate(state0, t_span, events=None, rtol=1e-9, atol=1e-11, max_step=0.05):
    """
    Integrate CR3BP from state0 over t_span (non-dimensional time).

    Parameters
    ----------
    state0 : array-like, shape (6,)
        [x, y, z, vx, vy, vz] in non-dimensional units.
    t_span : (t0, tf) in TU.
    events : list of callable or None.
        Each event(t, y) returns a scalar; integration stops when it crosses 0
        if event.terminal=True.
    rtol, atol : integrator tolerances.
    max_step : maximum step size (TU). Default 0.05 ≈ 45 min.

    Returns
    -------
    sol : OdeResult from scipy (sol.t, sol.y, sol.t_events, sol.y_events).
    """
    sol = solve_ivp(
        eom,
        t_span,
        state0,
        method="RK45",
        events=events,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )
    return sol


# ── Event factories ───────────────────────────────────────────────────────────

def make_earth_periapsis_event(r_capture_du=0.08):
    """
    Triggers at Earth periapsis (rdot = 0, inbound→outbound).

    Uses a smooth gating term so the event function is continuous:
        f = rdot + max(0, r − r_capture) * large_constant
    When r > r_capture the extra term keeps f positive (no trigger).
    When r < r_capture the extra term is 0 and f = rdot (triggers at periapsis).

    r_capture_du : gate radius in DU (default 0.08 ≈ 30 700 km from Earth).
    """
    K = 50.0   # large enough to keep f >> 0 when far

    def event(t, state):
        x, y, z, vx, vy, vz = state
        dx = x + MU
        dy = y
        dz = z
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        rdot = (dx*vx + dy*vy + dz*vz) / r
        gate = K * max(0.0, r - r_capture_du)
        return rdot + gate

    event.terminal  = True
    event.direction = 1   # rdot (+ gate) crosses 0 upward at periapsis
    return event


def make_moon_impact_event():
    """Stops integration if craft hits the Moon surface."""
    def event(t, state):
        x, y, z = state[:3]
        r2 = np.sqrt((x - 1 + MU)**2 + y**2 + z**2)
        return r2 - R_MOON_DU

    event.terminal = True
    event.direction = -1
    return event


def make_earth_impact_event():
    """Stops integration if craft hits the Earth surface."""
    def event(t, state):
        x, y, z = state[:3]
        r1 = np.sqrt((x + MU)**2 + y**2 + z**2)
        return r1 - R_EARTH_DU

    event.terminal = True
    event.direction = -1
    return event


def make_escape_event(r_max=5.0):
    """Stops integration if craft escapes the Earth–Moon system (r > r_max DU)."""
    def event(t, state):
        x, y, z = state[:3]
        r = np.sqrt(x**2 + y**2 + z**2)
        return r_max - r

    event.terminal = True
    event.direction = -1
    return event


def dist_from_earth(state):
    x, y, z = state[:3]
    return np.sqrt((x + MU)**2 + y**2 + z**2)


def dist_from_moon(state):
    x, y, z = state[:3]
    return np.sqrt((x - 1 + MU)**2 + y**2 + z**2)
