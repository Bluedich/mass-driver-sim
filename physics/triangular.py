"""
Quasi-periodic libration orbits about the triangular points (L4 / L5) of the
Earth–Moon CR3BP.

Background
----------
The equilateral points L4 and L5 are *linearly stable* for the Earth–Moon mass
ratio (μ ≈ 0.0122 is well below the Routh critical value 0.0385), so the
motion in their neighbourhood is bounded and quasi-periodic — a superposition of
two in-plane modes:

  * a **short-period** mode (period ≈ 28 d, close to the synodic month), and
  * a **long-period** mode (period ≈ 90 d).

In the real Earth–Moon–Sun system the long-period libration is the mode that the
Sun resonantly drives, and operational "Sun-synchronized" station-keeping orbits
about L4/L5 ride this long-period libration so that the vehicle keeps a fixed
orientation relative to the Sun.  This pure CR3BP model contains no Sun, so we
represent such a destination *geometrically*: a quasi-periodic torus built from
the long-period mode (with a small short-period ripple) at a chosen libration
amplitude.  The label "Sun-synchronized" names the intended real-world regime,
not a term of the idealised model.

Construction
------------
The two in-plane oscillation modes are obtained by numerically linearising the
CR3BP equations of motion at the triangular point and taking the two
purely-imaginary eigen-pairs (the out-of-plane z–mode, eigenvalue ±i, is
discarded — these orbits are planar).  An initial state is seeded from a scaled
combination of the mode eigenvectors and then propagated in the *full nonlinear*
model, so the stored curve is a true CR3BP trajectory, not a linear
approximation.
"""

import numpy as np

from .cr3bp import MU, DU_KM, eom, propagate, lagrange_points

# ── Triangular-point positions (DU) ───────────────────────────────────────────

_LP = {p[3]: np.array(p[:3]) for p in lagrange_points()}
L4 = _LP["L4"]          # (0.5 − μ, +√3/2, 0)
L5 = _LP["L5"]          # (0.5 − μ, −√3/2, 0)


# ── Linearised in-plane modes ─────────────────────────────────────────────────

def _jacobian(state):
    """6×6 Jacobian of the CR3BP EOM at `state` (central differences)."""
    state = np.asarray(state, dtype=float)
    n = state.size
    A = np.zeros((n, n))
    eps = 1e-6
    for j in range(n):
        sp = state.copy(); sp[j] += eps
        sm = state.copy(); sm[j] -= eps
        A[:, j] = (np.asarray(eom(0.0, sp)) - np.asarray(eom(0.0, sm))) / (2 * eps)
    return A


def _inplane_modes(point):
    """
    Long- and short-period in-plane eigen-modes at a triangular point.

    Returns (w_long, freq_long, w_short, freq_short) where each ``w`` is the
    complex 6-vector eigenvector of the eigenvalue +iω, and ``freq`` is ω
    (rad/TU).  Modes are identified as the two purely-imaginary eigen-pairs whose
    eigenvectors are essentially planar (negligible z / vz content); the
    out-of-plane z-mode (ω = 1) is excluded.
    """
    A = _jacobian(np.array([point[0], point[1], point[2], 0.0, 0.0, 0.0]))
    vals, vecs = np.linalg.eig(A)

    modes = []   # (freq, eigenvector) for each in-plane mode with Im(λ) > 0
    for lam, w in zip(vals, vecs.T):
        if lam.imag <= 1e-9:
            continue                          # take one representative per pair
        # In-plane modes have negligible out-of-plane (z, vz) content.
        planar = abs(w[2]) + abs(w[5])
        if planar > 1e-6 * (abs(w[0]) + abs(w[1]) + abs(w[3]) + abs(w[4])):
            continue                          # out-of-plane z-mode → skip
        modes.append((lam.imag, w))

    if len(modes) != 2:
        raise RuntimeError(
            f"Expected 2 in-plane modes at triangular point, found {len(modes)}."
        )

    modes.sort(key=lambda m: m[0])            # ascending frequency
    (freq_long, w_long), (freq_short, w_short) = modes
    return w_long, freq_long, w_short, freq_short


def _mode_perturbation(w, amp_du, phase=0.0):
    """
    Real 6-vector state perturbation for eigen-mode `w`, scaled so the in-plane
    position excursion peaks at `amp_du` (DU) over one cycle.

    The physical (real) motion of a mode with eigenvalue +iω is
    Re(w · e^{i(ωt+phase)}); this returns that expression at t = 0 for the given
    phase, normalised by the mode's peak in-plane radius.
    """
    phis = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    # position(φ) = Re(w[:2] · e^{iφ})  → shape (720, 2)
    pos = np.real(w[:2][None, :] * np.exp(1j * phis)[:, None])
    r_max = np.sqrt((pos ** 2).sum(axis=1)).max()

    delta = np.real(w * np.exp(1j * phase))
    return (amp_du / r_max) * delta


# ── Orbit construction ────────────────────────────────────────────────────────

# A modest short-period component gives the stored curve its quasi-periodic
# texture without letting it dominate the long-period libration.
_SHORT_FRAC = 0.18
_N_LONG     = 2       # long-periods traced (sweeps out the torus band)
_N_PTS      = 2000


def _build_qpo(point, amp_du):
    """
    One quasi-periodic libration orbit about `point` at long-period in-plane
    amplitude `amp_du` (DU).

    Returns (T_long, states) with states shape (_N_PTS, 6), a full nonlinear
    CR3BP trajectory over _N_LONG long-periods.
    """
    w_long, freq_long, w_short, _ = _inplane_modes(point)

    s0 = np.array([point[0], point[1], point[2], 0.0, 0.0, 0.0])
    state0 = (
        s0
        + _mode_perturbation(w_long,  amp_du)
        + _mode_perturbation(w_short, _SHORT_FRAC * amp_du)
    )

    T_long  = 2 * np.pi / freq_long
    T_total = _N_LONG * T_long
    t_eval  = np.linspace(0.0, T_total, _N_PTS, endpoint=False)

    sol = propagate(
        state0, (0.0, T_total * (1.0 + 1e-6)),
        t_eval=t_eval, rtol=1e-11, atol=1e-13, max_step=T_long / 400,
    )
    return T_long, sol.y.T


def _build_qpos(point, amp_km_list, point_label):
    """
    Quasi-periodic libration orbits about `point` for each amplitude in
    `amp_km_list` (long-period in-plane excursion, km).

    Returns
    -------
    list of dict, each with keys:
        'amp_km' : int
        'T'      : float   (long-period, TU)
        'states' : ndarray, shape (_N_PTS, 6)
    """
    out = []
    for amp_km in sorted(amp_km_list):
        T, states = _build_qpo(point, amp_km / DU_KM)
        out.append({"amp_km": int(amp_km), "T": float(T), "states": states})
    return out


def build_l4_qpos(amp_km_list=None):
    """Quasi-periodic libration orbits about L4 (default amplitudes)."""
    if amp_km_list is None:
        amp_km_list = [20_000, 40_000, 60_000]
    return _build_qpos(L4, amp_km_list, "L4")


def build_l5_qpos(amp_km_list=None):
    """Quasi-periodic libration orbits about L5 (default amplitudes)."""
    if amp_km_list is None:
        amp_km_list = [20_000, 40_000, 60_000]
    return _build_qpos(L5, amp_km_list, "L5")


# ── Approach event (arbitrary-centre capture sphere) ──────────────────────────

def make_point_approach_event(center, r_threshold):
    """
    Terminal event that fires when the spacecraft enters a sphere of radius
    `r_threshold` [DU] centred on the 3-vector `center` — the off-axis analogue
    of physics.halo.make_approach_event, needed because L4/L5 do not lie on the
    x-axis.  direction = -1 (distance decreasing → approaching).
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    def event(t, state):
        x, y, z = state[:3]
        return np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) - r_threshold

    event.terminal  = True
    event.direction = -1
    return event
