"""
Numba-JIT'd CR3BP propagator — a drop-in fast path for ``physics.cr3bp.propagate``.

The whole app's cost is thousands of independent CR3BP integrations, each driven
by scipy ``solve_ivp(RK45)`` with a *pure-Python* right-hand side and Python event
closures (see ``physics/cr3bp.py``).  This module removes that interpreter overhead:
a self-contained ``@njit`` adaptive Dormand-Prince RK45 with the dynamics inlined
and event detection done in-kernel.

Faithfulness to scipy
---------------------
The stepper reproduces scipy's ``RK45`` exactly — same Dormand-Prince Butcher
tableau, the same error norm ``rms((E·K)·h / (atol + rtol·max(|y|,|y_new|)))``,
the same elementary step-size controller (``SAFETY/MIN_FACTOR/MAX_FACTOR`` and the
``step_rejected`` clamp), and a port of ``select_initial_step``.  Events reproduce
scipy's ``find_active_events`` + ``handle_events``: after each accepted step, every
event's signed value is checked for a direction-matching sign change and the
crossing state is located on the quartic dense-output interpolant, with the
earliest terminal crossing winning.

Every terminal event in this codebase is a signed sphere crossing
``g(state) = ‖pos − (cx,cy,cz)‖ − radius`` (see the ``.sphere`` attributes on the
event factories), so a single generic kernel handles all destinations via an
``events`` array of shape ``(n_ev, 5) = [cx, cy, cz, radius, direction]``.

If numba is unavailable the module still imports (``@njit`` degrades to a no-op)
and ``HAVE_NUMBA`` is False, so callers fall back to scipy.
"""

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                       # numba missing or broken → scipy fallback
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        # Support both @njit and @njit(cache=True) usages as a no-op decorator.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn
        return _wrap

from .cr3bp import MU  # plain float — baked into the compiled kernel

# ── Dormand-Prince RK45 tableau (verbatim from scipy/integrate/_ivp/rk.py) ─────
_C45 = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0])
_A45 = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1/5, 0.0, 0.0, 0.0, 0.0],
    [3/40, 9/40, 0.0, 0.0, 0.0],
    [44/45, -56/15, 32/9, 0.0, 0.0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0.0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
])
_B45 = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84])
_E45 = np.array([-71/57600, 0.0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])
_P45 = np.array([
    [1.0, -8048581381/2820520608, 8663915743/2820520608, -12715105075/11282082432],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 131558114200/32700410799, -68118460800/10900136933, 87487479700/32700410799],
    [0.0, -1754552775/470086768, 14199869525/1410260304, -10690763975/1880347072],
    [0.0, 127303824393/49829197408, -318862633887/49829197408, 701980252875/199316789632],
    [0.0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
    [0.0, 40617522/29380423, -110615467/29380423, 69997945/29380423],
])

_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0
_ERR_EXPONENT = -1.0 / 5.0       # -1 / (error_estimator_order + 1), order 4 → -1/5


@njit(cache=True)
def _rhs(s, out):
    """CR3BP equations of motion (autonomous), written into ``out`` (len 6)."""
    x = s[0]; y = s[1]; z = s[2]; vx = s[3]; vy = s[4]; vz = s[5]
    r1 = ((x + MU) * (x + MU) + y * y + z * z) ** 0.5
    r2 = ((x - 1.0 + MU) * (x - 1.0 + MU) + y * y + z * z) ** 0.5
    c1 = (1.0 - MU) / (r1 * r1 * r1)
    c2 = MU / (r2 * r2 * r2)
    out[0] = vx
    out[1] = vy
    out[2] = vz
    out[3] = 2.0 * vy + x - c1 * (x + MU) - c2 * (x - 1.0 + MU)
    out[4] = -2.0 * vx + y - c1 * y - c2 * y
    out[5] = -c1 * z - c2 * z


@njit(cache=True)
def _rms(v):
    """RMS norm, matching scipy.integrate._ivp.common.norm."""
    s = 0.0
    for i in range(v.shape[0]):
        s += v[i] * v[i]
    return (s / v.shape[0]) ** 0.5


@njit(cache=True)
def _select_initial_step(y0, f0, t0, tf, direction, max_step, rtol, atol):
    """Port of scipy select_initial_step for error_estimator_order = 4."""
    interval_length = abs(tf - t0)
    if interval_length == 0.0:
        return 0.0

    n = y0.shape[0]
    scale = np.empty(n)
    d0v = np.empty(n)
    d1v = np.empty(n)
    for i in range(n):
        scale[i] = atol + abs(y0[i]) * rtol
        d0v[i] = y0[i] / scale[i]
        d1v[i] = f0[i] / scale[i]
    d0 = _rms(d0v)
    d1 = _rms(d1v)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    if h0 > interval_length:
        h0 = interval_length

    y1 = np.empty(n)
    for i in range(n):
        y1[i] = y0[i] + h0 * direction * f0[i]
    f1 = np.empty(n)
    _rhs(y1, f1)

    d2v = np.empty(n)
    for i in range(n):
        d2v[i] = (f1[i] - f0[i]) / scale[i]
    d2 = _rms(d2v) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = 1e-6 if 1e-6 > h0 * 1e-3 else h0 * 1e-3
    else:
        m = d1 if d1 > d2 else d2
        h1 = (0.01 / m) ** (1.0 / 5.0)

    h = 100.0 * h0
    if h1 < h:
        h = h1
    if interval_length < h:
        h = interval_length
    if max_step < h:
        h = max_step
    return h


@njit(cache=True)
def _rk_step(y, f, h, K):
    """One Dormand-Prince step from (y, f) with step h. Fills K (7,6). Returns y_new."""
    n = y.shape[0]
    for j in range(n):
        K[0, j] = f[j]
    ytmp = np.empty(n)
    # Stages 1..5
    for s in range(1, 6):
        for j in range(n):
            acc = 0.0
            for m in range(s):
                acc += _A45[s, m] * K[m, j]
            ytmp[j] = y[j] + h * acc
        _rhs(ytmp, K[s])
    # y_new = y + h * B·K[:6]
    y_new = np.empty(n)
    for j in range(n):
        acc = 0.0
        for m in range(6):
            acc += _B45[m] * K[m, j]
        y_new[j] = y[j] + h * acc
    # FSAL last stage
    f_new = np.empty(n)
    _rhs(y_new, f_new)
    for j in range(n):
        K[6, j] = f_new[j]
    return y_new, f_new


@njit(cache=True)
def _error_norm(K, h, y, y_new, rtol, atol):
    n = y.shape[0]
    en = np.empty(n)
    for j in range(n):
        err = 0.0
        for m in range(7):
            err += _E45[m] * K[m, j]
        err *= h
        ay = abs(y[j])
        ayn = abs(y_new[j])
        scale = atol + (ay if ay > ayn else ayn) * rtol
        en[j] = err / scale
    return _rms(en)


@njit(cache=True)
def _interp(y_old, h, Q, x, out):
    """Quartic dense-output: out = y_old + h * Q·[x, x^2, x^3, x^4]."""
    n = y_old.shape[0]
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    for j in range(n):
        out[j] = y_old[j] + h * (Q[j, 0] * x + Q[j, 1] * x2 + Q[j, 2] * x3 + Q[j, 3] * x4)


@njit(cache=True)
def _g_sphere(state, cx, cy, cz, radius):
    dx = state[0] - cx
    dy = state[1] - cy
    dz = state[2] - cz
    return (dx * dx + dy * dy + dz * dz) ** 0.5 - radius


@njit(cache=True)
def _solve_event_x(y_old, h, Q, cx, cy, cz, radius, g0):
    """Bisection for the sphere-crossing x in [0,1] on the dense interpolant.

    g0 is g at x=0 (i.e. at y_old). Assumes a sign change over [0,1].
    """
    tmp = np.empty(6)
    xa = 0.0
    xb = 1.0
    ga = g0
    # g at x=1
    _interp(y_old, h, Q, 1.0, tmp)
    gb = _g_sphere(tmp, cx, cy, cz, radius)
    # Bisection to tight x tolerance (superlinear brentq not needed — this runs
    # once per trajectory at termination).
    for _ in range(80):
        xm = 0.5 * (xa + xb)
        _interp(y_old, h, Q, xm, tmp)
        gm = _g_sphere(tmp, cx, cy, cz, radius)
        if gm == 0.0 or (xb - xa) < 1e-15:
            return xm
        if (ga < 0.0) != (gm < 0.0):
            xb = xm
            gb = gm
        else:
            xa = xm
            ga = gm
    return 0.5 * (xa + xb)


@njit(cache=True)
def integrate_one(state0, t0, tf, events, rtol, atol, max_step,
                  t_out, y_out, y_cross):
    """Integrate the CR3BP from state0 over [t0, tf] with terminal sphere events.

    Parameters
    ----------
    state0 : (6,) initial state.
    events : (n_ev, 5) array of [cx, cy, cz, radius, direction]; row 0 is the
        destination "success" event (matching the events=[dest, ...] convention).
    t_out, y_out : preallocated (max_samples,) and (max_samples, 6) path buffers.
    y_cross : preallocated (6,) buffer, filled with the crossing state on a hit.

    Returns
    -------
    (event_idx, t_cross, n_samples) : event_idx is the 0-based index of the event
        that fired (or -1 if the integration ran to tf with no event). n_samples
        is the number of stored path points.
    """
    n = 6
    n_ev = events.shape[0]
    max_samples = t_out.shape[0]
    direction = 1.0 if tf >= t0 else -1.0

    y = state0.copy()
    f = np.empty(n)
    _rhs(y, f)

    h_abs = _select_initial_step(y, f, t0, tf, direction, max_step, rtol, atol)

    K = np.empty((7, n))
    Q = np.empty((n, 4))
    y_old = np.empty(n)

    # Seed event values at t0 (scipy seeds g = event(t0, y0) before the loop).
    g_old = np.empty(n_ev)
    for e in range(n_ev):
        g_old[e] = _g_sphere(y, events[e, 0], events[e, 1], events[e, 2], events[e, 3])

    # Store initial sample.
    t = t0
    n_samp = 0
    if max_samples > 0:
        t_out[0] = t0
        for j in range(n):
            y_out[0, j] = y[j]
        n_samp = 1

    while True:
        # ── Reached the end of the interval? ─────────────────────────────────
        if (tf - t) * direction <= 0.0:
            return -1, t, n_samp

        min_step = 10.0 * abs(np.nextafter(t, direction * np.inf) - t)
        if h_abs > max_step:
            h_abs = max_step
        if h_abs < min_step:
            h_abs = min_step

        step_accepted = False
        step_rejected = False
        y_new = y
        f_new = f
        h = h_abs * direction

        while not step_accepted:
            if h_abs < min_step:
                # Cannot make progress; bail out at current t.
                return -1, t, n_samp
            h = h_abs * direction
            t_new = t + h
            if (t_new - tf) * direction > 0.0:
                t_new = tf
            h = t_new - t
            h_abs = abs(h)

            y_new, f_new = _rk_step(y, f, h, K)
            err = _error_norm(K, h, y, y_new, rtol, atol)

            if err < 1.0:
                if err == 0.0:
                    factor = _MAX_FACTOR
                else:
                    factor = _SAFETY * err ** _ERR_EXPONENT
                    if factor > _MAX_FACTOR:
                        factor = _MAX_FACTOR
                if step_rejected and factor > 1.0:
                    factor = 1.0
                h_abs *= factor
                step_accepted = True
            else:
                factor = _SAFETY * err ** _ERR_EXPONENT
                if factor < _MIN_FACTOR:
                    factor = _MIN_FACTOR
                h_abs *= factor
                step_rejected = True

        t_new = t + h
        for j in range(n):
            y_old[j] = y[j]

        # ── Event detection over the accepted step ───────────────────────────
        # g_new at the step end.
        best_x = 2.0
        best_e = -1
        need_Q = True
        g_new = np.empty(n_ev)
        for e in range(n_ev):
            g_new[e] = _g_sphere(y_new, events[e, 0], events[e, 1], events[e, 2], events[e, 3])
            go = g_old[e]
            gn = g_new[e]
            d = events[e, 4]
            up = (go <= 0.0) and (gn >= 0.0)
            down = (go >= 0.0) and (gn <= 0.0)
            active = (up and d > 0.0) or (down and d < 0.0) or ((up or down) and d == 0.0)
            if active:
                if need_Q:
                    # Q = K.T · P  (n×4)
                    for j in range(n):
                        for c in range(4):
                            acc = 0.0
                            for m in range(7):
                                acc += K[m, j] * _P45[m, c]
                            Q[j, c] = acc
                    need_Q = False
                xr = _solve_event_x(y_old, h, Q, events[e, 0], events[e, 1],
                                    events[e, 2], events[e, 3], go)
                if xr < best_x:
                    best_x = xr
                    best_e = e

        if best_e >= 0:
            # Terminal crossing — set state to the interpolated crossing and stop.
            t_cross = t + best_x * h
            _interp(y_old, h, Q, best_x, y_cross)
            if n_samp < max_samples:
                t_out[n_samp] = t_cross
                for j in range(n):
                    y_out[n_samp, j] = y_cross[j]
                n_samp += 1
            return best_e, t_cross, n_samp

        # Advance and store the accepted step.
        t = t_new
        y = y_new
        f = f_new
        for e in range(n_ev):
            g_old[e] = g_new[e]
        if n_samp < max_samples:
            t_out[n_samp] = t
            for j in range(n):
                y_out[n_samp, j] = y[j]
            n_samp += 1


class _FastSol:
    """Duck-typed stand-in for scipy's OdeResult (only the attributes callers read)."""
    __slots__ = ("t", "y", "t_events", "y_events")

    def __init__(self, t, y, t_events, y_events):
        self.t = t
        self.y = y
        self.t_events = t_events
        self.y_events = y_events


def _events_to_array(events):
    """Stack event factories' ``.sphere`` tuples into an (n_ev, 5) float array.

    Returns None if any event lacks ``.sphere`` (→ caller must use scipy).
    """
    rows = []
    for ev in events:
        sph = getattr(ev, "sphere", None)
        if sph is None:
            return None
        rows.append(sph)
    return np.array(rows, dtype=np.float64)


def propagate_fast(state0, t_span, events, rtol, atol, max_step, max_samples=20000):
    """scipy-``propagate``-compatible fast path. Returns a ``_FastSol``.

    ``events`` is the same list of event *factories* the destinations pass; each
    must carry a ``.sphere`` attribute (checked by the caller via
    ``_events_to_array``). Row 0 is treated as the destination success event, so
    ``sol.t_events[0]`` / ``sol.y_events[0]`` behave like scipy's.
    """
    events_arr = _events_to_array(events)
    if events_arr is None:
        raise ValueError("propagate_fast requires all events to expose .sphere")

    t0, tf = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(state0, dtype=np.float64).copy()

    t_out = np.empty(max_samples, dtype=np.float64)
    y_out = np.empty((max_samples, 6), dtype=np.float64)
    y_cross = np.empty(6, dtype=np.float64)

    ev_idx, t_cross, n_samp = integrate_one(
        y0, t0, tf, events_arr, float(rtol), float(atol), float(max_step),
        t_out, y_out, y_cross,
    )

    t = t_out[:n_samp].copy()
    y = y_out[:n_samp].T.copy()          # shape (6, n) to match scipy sol.y

    n_ev = len(events)
    t_events = [np.empty(0) for _ in range(n_ev)]
    y_events = [np.empty((0, 6)) for _ in range(n_ev)]
    if ev_idx >= 0:
        t_events[ev_idx] = np.array([t_cross])
        y_events[ev_idx] = y_cross.reshape(1, 6).copy()

    return _FastSol(t=t, y=y, t_events=t_events, y_events=y_events)


def warmup():
    """Force JIT compilation (or load the on-disk cache) off the timing-critical
    path. Called once per worker process from the pool initializer so the first
    real propagation isn't charged the ~1–2 s compile latency. No-op if numba
    is unavailable."""
    if not HAVE_NUMBA:
        return
    state0 = np.array([0.9, 0.0, 0.0, 0.0, 0.5, 0.0])
    events = np.array([[1.0 - MU, 0.0, 0.0, 0.05, -1.0]])
    t_out = np.empty(16, dtype=np.float64)
    y_out = np.empty((16, 6), dtype=np.float64)
    y_cross = np.empty(6, dtype=np.float64)
    integrate_one(state0, 0.0, 0.05, events, 1e-7, 1e-9, 0.01,
                  t_out, y_out, y_cross)
