"""
L1 planar Lyapunov orbit family for the Earth-Moon CR3BP.

A Lyapunov orbit is the planar (z = 0) periodic orbit around L1 — the in-plane
analogue of the halo family.  Initial conditions are found by differential
correction with continuation from small to large x-amplitude.

The orbit is parametrised by its x-amplitude Ax (the +x excursion of the start
point from L1): the initial state is [L1_X + Ax, 0, 0, 0, vy0, 0] and the single
free variable vy0 is solved so that vx = 0 at the y = 0 half-period crossing.
The out-of-plane components (z, vz) stay identically zero by planar symmetry.

The linearised-frequency constants are shared with physics/halo.py.
"""

import numpy as np
from scipy.optimize import brentq

from .cr3bp import DU_KM, propagate
from .halo import L1_X, K, WP, L2_X, K2, WP2, _T_GATE


def _half_period_vx(vy0, x0):
    """
    vx at the y = 0 half-period crossing for the planar orbit starting at
    [x0, 0, 0, 0, vy0, 0] (vy0 < 0).

    y first goes negative, then returns to 0 going upward at T/2; for a
    periodic planar orbit vx must vanish there.  This residual is smooth and
    monotonically increasing in vy0 along the Lyapunov branch, so a 1-D
    bracketing root finder (brentq) locates the orbit robustly — unlike
    fsolve, whose finite-difference Jacobian wanders onto a spurious branch
    that loops past the Moon.

    Returns +/- 1e6 if no upward crossing is found (keeps the bracket search
    well-signed).
    """
    state0 = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])

    # Phase 1: integrate past t=0 so the y=0 event does not trigger immediately.
    sg = propagate(state0, (0.0, _T_GATE),
                   rtol=1e-9, atol=1e-11, max_step=0.02)

    # Phase 2: integrate to the y=0 re-crossing going upward (half-period).
    def ev(t, s):
        return s[1]
    ev.terminal  = True
    ev.direction = 1

    sh = propagate(sg.y[:, -1], (_T_GATE, 6.0),
                   events=[ev], rtol=1e-9, atol=1e-11, max_step=0.02)

    if not sh.t_events[0].size:
        return 1e6 if vy0 < 0 else -1e6

    return float(sh.y_events[0][0][3])      # vx at the crossing


def _find_lyapunov(x0, vy0_seed):
    """
    Differential correction for one planar Lyapunov orbit at fixed x0.

    Brackets the root of _half_period_vx around vy0_seed (residual is monotone
    in vy0) then refines with brentq.  Returns vy0 if found, else None.
    """
    f_seed = _half_period_vx(vy0_seed, x0)
    if not np.isfinite(f_seed):
        return None

    # Expand outward from the seed until the residual changes sign.
    step  = 0.1 * abs(vy0_seed) + 1e-4
    direction = -1.0 if f_seed > 0 else 1.0   # root lies toward more-negative vy0
                                              # when residual is positive
    lo, flo = vy0_seed, f_seed
    hi, fhi = vy0_seed, f_seed
    for _ in range(80):
        nxt  = (lo if direction < 0 else hi) + direction * step
        fnxt = _half_period_vx(nxt, x0)
        if not np.isfinite(fnxt):
            return None
        if direction < 0:
            hi, fhi = lo, flo
            lo, flo = nxt, fnxt
        else:
            lo, flo = hi, fhi
            hi, fhi = nxt, fnxt
        if flo < 0.0 < fhi:
            try:
                root = brentq(_half_period_vx, lo, hi, args=(x0,),
                              xtol=1e-12, rtol=1e-12)
            except Exception:
                return None
            if abs(_half_period_vx(root, x0)) > 1e-6:
                return None
            return float(root)

    return None


def _sample_orbit(x0, vy0, n_pts=500):
    """
    Integrate one full planar Lyapunov period and return uniformly-sampled
    states.

    Returns (T_full, states) where states has shape (n_pts, 6).
    """
    state0 = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])

    sg = propagate(state0, (0.0, _T_GATE), rtol=1e-10, atol=1e-12, max_step=0.005)

    def ev(t, s):
        return s[1]
    ev.terminal  = True
    ev.direction = 1

    sh = propagate(sg.y[:, -1], (_T_GATE, 6.0),
                   events=[ev], rtol=1e-10, atol=1e-12, max_step=0.005)
    T_half = sh.t_events[0][0]
    T_full = 2.0 * T_half

    t_eval = np.linspace(0.0, T_full, n_pts + 1)[:-1]
    sol = propagate(state0, (0.0, T_full * (1.0 + 1e-6)),
                    t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=T_full / 300)

    return T_full, sol.y.T


# ── Public API ────────────────────────────────────────────────────────────────

def _build_lyapunovs(ax_km_list, Lx, wp, k, point_label):
    """
    Compute planar Lyapunov orbits about a collinear point for each Ax value.

    Uses continuation: bootstraps at a small amplitude (where the linear-theory
    guess is accurate) then steps upward in 500 km increments, re-solving vy0 at
    each fixed x0 = Lx + Ax.

    Parameters
    ----------
    ax_km_list  : list of int    in-plane x-amplitudes [km]
    Lx, wp, k   : float          libration-point linear constants
    point_label : str            used only in the error/warning messages

    Returns
    -------
    list of dict, each with keys 'ax_km', 'T', 'states' (see build_l1_lyapunovs).
    """
    ax_km_sorted = sorted(ax_km_list)
    ax_km_set    = set(ax_km_sorted)
    ax_max_km    = ax_km_sorted[-1]

    # ── Bootstrap at a small amplitude where linear theory is accurate ───────
    # (continuing up from here keeps the corrector on the Lyapunov branch).
    AX_BOOT_KM = min(2_000, ax_km_sorted[0])
    ax_boot    = AX_BOOT_KM / DU_KM
    x0_boot    = Lx + ax_boot
    vy0_guess  = -k * ax_boot * wp            # negative so y first goes negative

    vy0 = _find_lyapunov(x0_boot, vy0_guess)
    if vy0 is None:
        raise RuntimeError(
            f"{point_label} Lyapunov bootstrap at Ax={AX_BOOT_KM} km failed to "
            "converge.  Check that physics/cr3bp.py is importable and MU is "
            "correct."
        )

    results = {}
    if AX_BOOT_KM in ax_km_set:
        T, states = _sample_orbit(x0_boot, vy0)
        results[AX_BOOT_KM] = {"ax_km": AX_BOOT_KM, "T": T, "states": states}

    # ── Continuation: step from the bootstrap amplitude toward ax_max_km ─────
    AX_STEP_KM = 500
    ax_km_cur  = AX_BOOT_KM

    while ax_km_cur < ax_max_km:
        ax_km_next = ax_km_cur + AX_STEP_KM
        x0_next    = Lx + ax_km_next / DU_KM

        vy0_next = _find_lyapunov(x0_next, vy0)
        if vy0_next is None:
            import warnings
            warnings.warn(
                f"{point_label} Lyapunov continuation failed at Ax={ax_km_next} "
                "km; stopping continuation."
            )
            break

        vy0       = vy0_next
        ax_km_cur = ax_km_next

        if ax_km_next in ax_km_set:
            T, states = _sample_orbit(x0_next, vy0)
            results[ax_km_next] = {"ax_km": ax_km_next, "T": T, "states": states}

    return [results[ax] for ax in ax_km_sorted if ax in results]


def build_l1_lyapunovs(ax_km_list=None):
    """
    Planar L1 Lyapunov orbits for each Ax value in ax_km_list (default
    [5000, 10000, 20000, 30000]).

    Returns
    -------
    list of dict, each with keys:
        'ax_km'  : int
        'T'      : float   (non-dim full period, TU)
        'states' : ndarray, shape (N_PTS, 6) — one full period uniformly sampled
    """
    if ax_km_list is None:
        ax_km_list = [5_000, 10_000, 20_000, 30_000]
    return _build_lyapunovs(ax_km_list, L1_X, WP, K, point_label="L1")


def build_l2_lyapunovs(ax_km_list=None):
    """
    Planar L2 Lyapunov orbits for each Ax value in ax_km_list (default
    [5000, 10000, 15000, 20000]).

    The L2 Lyapunov family stays clear of the Moon up to ~20 000 km; beyond that
    the orbits begin to wrap around the Moon, so the default amplitudes stop
    there.  Same return shape as build_l1_lyapunovs.
    """
    if ax_km_list is None:
        ax_km_list = [5_000, 10_000, 15_000, 20_000]
    return _build_lyapunovs(ax_km_list, L2_X, WP2, K2, point_label="L2")
