"""
L1 halo orbit family for the Earth-Moon CR3BP.

Computes northern halo orbit initial conditions via differential correction
(continuation from small to large Az amplitude).  The module-level constants
(L1 position, linearised frequencies) are evaluated once at import time.
"""

import numpy as np
from scipy.optimize import fsolve

from .cr3bp import MU, DU_KM, VU_KMS, lagrange_points, propagate

# ── Linearised dynamics at L1 ─────────────────────────────────────────────────

def _collinear_constants(idx):
    """
    Linearised dynamics at a collinear libration point.

    idx : 0 = L1, 1 = L2 (matches lagrange_points() ordering).

    The in-plane / out-of-plane frequencies and the |vy|/|vx| amplitude ratio K
    are only used to seed the differential corrector, so the same c2 expression
    (Earth at distance 1+gamma) serves both points to within the corrector's
    basin of attraction.
    """
    pts = lagrange_points()
    Lx    = pts[idx][0]
    gamma = abs((1.0 - MU) - Lx)             # distance to the Moon [DU]

    c2 = MU / gamma**3 + (1.0 - MU) / (1.0 + gamma)**3

    disc = np.sqrt(9.0 * c2**2 - 8.0 * c2)
    wp   = np.sqrt((2.0 - c2 + disc) / 2.0)  # in-plane oscillation freq [rad/TU]
    wz   = np.sqrt(c2)                         # out-of-plane freq [rad/TU]
    k    = (wp**2 + 1.0 + 2.0 * c2) / (2.0 * wp)  # |vy|/|vx| amplitude ratio

    return Lx, gamma, c2, wp, wz, k


L1_X, GAMMA1, C2,    WP,  WZ,  K  = _collinear_constants(0)
L2_X, GAMMA2, C2_L2, WP2, WZ2, K2 = _collinear_constants(1)

# ── Differential corrector internals ─────────────────────────────────────────

# Gate time (TU): skip past t=0 before enabling the y=0 re-crossing event.
# All halo periods near L1 are > 1.5 TU, so 0.5 TU is safely inside the orbit.
_T_GATE = 0.5


def _half_period_residuals(free_vars, az_du):
    """
    Residuals for the differential corrector.

    For a northern halo orbit starting at [x0, 0, az_du, 0, vy0, 0]:
      - vy0 < 0 → y first goes negative
      - At T/2, y returns to 0 going upward: vx=0 and vz=0 (symmetry condition)

    Returns [vx(T/2), vz(T/2)] — both should be zero for a periodic orbit.
    """
    x0, vy0 = free_vars
    state0 = np.array([x0, 0.0, az_du, 0.0, vy0, 0.0])

    # Phase 1: integrate T_GATE TU with no events (avoids triggering at t=0)
    sg = propagate(state0, (0.0, _T_GATE),
                   rtol=1e-9, atol=1e-11, max_step=0.02)

    # Phase 2: integrate until y=0 crossing going upward (half-period)
    def ev(t, s):
        return s[1]          # y = 0
    ev.terminal  = True
    ev.direction = 1         # upward: vy0<0 so y first goes −, returns + at T/2

    sh = propagate(sg.y[:, -1], (_T_GATE, 6.0),
                   events=[ev], rtol=1e-9, atol=1e-11, max_step=0.02)

    if not sh.t_events[0].size:
        return [1e6, 1e6]   # half-period crossing not found

    s = sh.y_events[0][0]
    return [s[3], s[5]]     # vx, vz at y=0 half-period crossing


def _find_halo(az_du, x0_guess, vy0_guess):
    """
    Differential correction for one northern halo orbit.

    Returns (x0, vy0) if converged, else None.
    The full initial state is [x0, 0, az_du, 0, vy0, 0].
    """
    try:
        sol, _, ier, _ = fsolve(
            _half_period_residuals,
            [x0_guess, vy0_guess],
            args=(az_du,),
            xtol=1e-10,
            full_output=True,
        )
    except Exception:
        return None

    if ier != 1:
        return None

    # Verify residuals are genuinely small
    res = _half_period_residuals(sol, az_du)
    if max(abs(res[0]), abs(res[1])) > 1e-6:
        return None

    return float(sol[0]), float(sol[1])


def _sample_orbit(x0, vy0, az_du, n_pts=500):
    """
    Integrate one full northern halo orbit period and return uniformly-sampled
    states.

    Returns (T_full, states) where states has shape (n_pts, 6).
    """
    state0 = np.array([x0, 0.0, az_du, 0.0, vy0, 0.0])

    # Determine T_half using the same two-phase approach
    sg = propagate(state0, (0.0, _T_GATE), rtol=1e-10, atol=1e-12, max_step=0.005)

    def ev(t, s):
        return s[1]
    ev.terminal  = True
    ev.direction = 1

    sh = propagate(sg.y[:, -1], (_T_GATE, 6.0),
                   events=[ev], rtol=1e-10, atol=1e-12, max_step=0.005)
    # sh.t_events gives absolute times — no need to add _T_GATE again
    T_half = sh.t_events[0][0]
    T_full = 2.0 * T_half

    # Dense sample over exactly one period
    t_eval = np.linspace(0.0, T_full, n_pts + 1)[:-1]
    sol = propagate(state0, (0.0, T_full * (1.0 + 1e-6)),
                    t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=T_full / 300)

    return T_full, sol.y.T   # (n_pts, 6)


# ── Public API ────────────────────────────────────────────────────────────────

def _build_halos(az_km_list, Lx, wp, k, x_dir, point_label):
    """
    Compute northern halo orbits about a collinear point for each Az value.

    Uses continuation: bootstraps at Az=5 000 km (where linear theory is
    accurate) then steps upward in 500 km increments to reach each target.

    Parameters
    ----------
    az_km_list : list of int     out-of-plane amplitudes [km]
    Lx, wp, k  : float           libration-point linear constants
    x_dir      : +1 | -1         sign of the bootstrap x-excursion from Lx
                                 (L1 halos sit toward the Moon, L2 away from it)
    point_label: str             used only in the error/warning messages

    Returns
    -------
    list of dict, each with keys 'az_km', 'T', 'states' (see build_l1_halos).
    """
    az_km_sorted = sorted(az_km_list)
    az_km_set    = set(az_km_sorted)
    az_max_km    = az_km_sorted[-1]

    # ── Bootstrap at Az = 5 000 km via linear-theory guess ───────────────────
    AZ_BOOT_KM = 5_000
    az_boot    = AZ_BOOT_KM / DU_KM
    ax_guess   = 1.5 * az_boot                   # rough Ax ≈ 1.5 Az for halos
    x0_boot    = Lx + x_dir * ax_guess
    vy0_boot   = -k * ax_guess * wp              # negative for northern halo

    boot = _find_halo(az_boot, x0_boot, vy0_boot)
    if boot is None:
        raise RuntimeError(
            f"{point_label} halo bootstrap at Az={AZ_BOOT_KM} km failed to "
            "converge.  Check that physics/cr3bp.py is importable and MU is "
            "correct."
        )
    x0_cur, vy0_cur = boot

    results = {}
    if AZ_BOOT_KM in az_km_set:
        T, states = _sample_orbit(x0_cur, vy0_cur, az_boot)
        results[AZ_BOOT_KM] = {"az_km": AZ_BOOT_KM, "T": T, "states": states}

    # ── Continuation: step from 5 000 km toward az_max_km ─────────────────────
    AZ_STEP_KM = 500
    az_km_cur  = AZ_BOOT_KM

    while az_km_cur < az_max_km:
        az_km_next = az_km_cur + AZ_STEP_KM
        az_next    = az_km_next / DU_KM

        sol = _find_halo(az_next, x0_cur, vy0_cur)
        if sol is None:
            import warnings
            warnings.warn(
                f"{point_label} halo continuation failed at Az={az_km_next} km; "
                "stopping continuation."
            )
            break

        x0_cur, vy0_cur = sol
        az_km_cur = az_km_next

        if az_km_next in az_km_set:
            T, states = _sample_orbit(x0_cur, vy0_cur, az_next)
            results[az_km_next] = {"az_km": az_km_next, "T": T, "states": states}

    return [results[az] for az in az_km_sorted if az in results]


def build_l1_halos(az_km_list=None):
    """
    Northern L1 halo orbits for each Az value in az_km_list (default
    [5000, 10000, 20000, 30000]).  See _build_halos for the return shape.
    """
    if az_km_list is None:
        az_km_list = [5_000, 10_000, 20_000, 30_000]
    return _build_halos(az_km_list, L1_X, WP, K, x_dir=+1, point_label="L1")


def build_l2_halos(az_km_list=None):
    """
    Northern L2 halo orbits for each Az value in az_km_list (default
    [5000, 10000, 20000, 30000]).  L2 sits beyond the Moon, so the bootstrap
    x-excursion points back toward it (x_dir=-1).
    """
    if az_km_list is None:
        az_km_list = [5_000, 10_000, 20_000, 30_000]
    return _build_halos(az_km_list, L2_X, WP2, K2, x_dir=-1, point_label="L2")


# ── Approach events (used by destinations/l{1,2}_*.py) ───────────────────────

def make_approach_event(center_x, r_threshold=0.09):
    """
    Terminal event that fires when the spacecraft enters a sphere of radius
    r_threshold [DU] centred on (center_x, 0, 0).

    r_threshold = 0.09 DU ≈ 34 600 km — encloses the target orbit set.
    direction = -1 (distance decreasing, i.e. approaching the point).
    """
    def event(t, state):
        x, y, z = state[:3]
        return np.sqrt((x - center_x)**2 + y**2 + z**2) - r_threshold

    event.terminal  = True
    event.direction = -1
    # Generic sphere-crossing form for the JIT integrator.
    event.sphere = (float(center_x), 0.0, 0.0, float(r_threshold), -1.0)
    return event


def make_l1_approach_event(r_threshold=0.09):
    """Approach sphere centred on L1."""
    return make_approach_event(L1_X, r_threshold)


def make_l2_approach_event(r_threshold=0.09):
    """Approach sphere centred on L2."""
    return make_approach_event(L2_X, r_threshold)
