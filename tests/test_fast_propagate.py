"""
Regression tests for the numba JIT trajectory fast path (physics/fast_propagate.py).

Every test compares the fast path against the scipy RK45 reference (forced with
MDS_DISABLE_FAST=1) over the same states, tolerances, and events, asserting:

  * identical reachability classification (an event fired or not),
  * crossing-state agreement to interpolation tolerance,
  * post-launch ΔV agreement well within 1e-4 km/s, and
  * Jacobi-constant conservation along a long fast trajectory.

If numba is unavailable the fast path degrades to the scipy path and these tests
become trivially true (fast == reference); they still exercise the pure-Python
integrator for correctness.
"""

import os

import numpy as np
import pytest

from physics import fast_propagate as fp
from physics.cr3bp import (
    propagate, jacobi,
    make_moon_impact_event, make_earth_impact_event, make_escape_event,
)
from physics.coordinates import initial_state, speed_du_per_tu
from physics.cr3bp import VU_KMS


def _events_and_tol(dest_id):
    if dest_id == "earth_leo_1200":
        from destinations.earth_leo import make_target_altitude_event
        ev = [make_target_altitude_event(), make_moon_impact_event(),
              make_earth_impact_event(), make_escape_event(r_max=3.5)]
        return ev, dict(rtol=1e-7, atol=1e-9, max_step=0.01), 8.0
    if dest_id == "l1_halo":
        from physics.halo import make_approach_event, L1_X
        ev = [make_approach_event(L1_X, 0.09), make_moon_impact_event(),
              make_earth_impact_event(), make_escape_event(r_max=3.5)]
        return ev, dict(rtol=1e-8, atol=1e-10, max_step=0.02), 12.0
    if dest_id == "l4_qpo":
        from physics.triangular import make_point_approach_event
        from physics.cr3bp import lagrange_points
        L4 = np.array(lagrange_points()[3][:3])
        ev = [make_point_approach_event(L4, 0.32), make_moon_impact_event(),
              make_earth_impact_event(), make_escape_event(r_max=3.5)]
        return ev, dict(rtol=1e-8, atol=1e-10, max_step=0.03), 30.0
    raise ValueError(dest_id)


def _sample_states(n=8):
    lats = np.linspace(-70, 70, n)
    lons = np.linspace(-170, 170, n)
    speeds = speed_du_per_tu(np.array([2.45, 2.55, 2.63, 2.72]))
    out = []
    for i in range(n):
        for az in (0, 90, 180, 270):
            for el in (0.0, 5.0):
                for v in speeds:
                    out.append(initial_state(lats[i], lons[i], az, el, v))
    return out


def _scipy_propagate(state0, tmax, events, tol):
    """Reference propagation forced onto scipy regardless of numba availability."""
    old = os.environ.get("MDS_DISABLE_FAST")
    os.environ["MDS_DISABLE_FAST"] = "1"
    try:
        return propagate(state0, (0.0, tmax), events=events, **tol)
    finally:
        if old is None:
            os.environ.pop("MDS_DISABLE_FAST", None)
        else:
            os.environ["MDS_DISABLE_FAST"] = old


@pytest.mark.parametrize("dest_id", ["earth_leo_1200", "l1_halo", "l4_qpo"])
def test_fast_matches_scipy(dest_id):
    events, tol, tmax = _events_and_tol(dest_id)
    states = _sample_states()

    n_cmp = 0
    reach_mismatch = 0
    worst_state = 0.0
    for s0 in states:
        ref = _scipy_propagate(s0, tmax, events, tol)
        fast = propagate(s0, (0.0, tmax), events=events, **tol)

        hit_ref = ref.t_events[0].size > 0
        hit_fast = fast.t_events[0].size > 0
        if hit_ref != hit_fast:
            reach_mismatch += 1
            continue
        n_cmp += 1
        if hit_ref:
            d = np.abs(ref.y_events[0][0] - fast.y_events[0][0]).max()
            worst_state = max(worst_state, d)

    assert reach_mismatch == 0, f"{dest_id}: {reach_mismatch} reachability mismatches"
    # Crossing state (6-vector, non-dim) must match to interpolation accuracy.
    assert worst_state < 1e-7, f"{dest_id}: max crossing-state diff {worst_state:.2e}"


def test_deltav_map_matches_scipy():
    """End-to-end: the per-tile ΔV array from compute_grid must match scipy."""
    from physics.tiling import generate_tiling
    from physics.optimizer import compute_grid
    from destinations.earth_leo import EARTH_LEO_1200

    sites = generate_tiling(24).sites

    old = os.environ.get("MDS_DISABLE_FAST")
    os.environ["MDS_DISABLE_FAST"] = "1"
    try:
        dv_ref = compute_grid(sites, EARTH_LEO_1200)[0]
    finally:
        if old is None:
            os.environ.pop("MDS_DISABLE_FAST", None)
        else:
            os.environ["MDS_DISABLE_FAST"] = old

    dv_fast = compute_grid(sites, EARTH_LEO_1200)[0]

    assert np.array_equal(np.isfinite(dv_ref), np.isfinite(dv_fast)), "reachability differs"
    both = np.isfinite(dv_ref) & np.isfinite(dv_fast)
    max_ddv = float(np.max(np.abs(dv_ref[both] - dv_fast[both]))) if both.any() else 0.0
    assert max_ddv < 1e-4, f"max|Δdv| = {max_ddv:.2e} km/s exceeds 1e-4"


def test_jacobi_conserved_long_arc():
    """A long L4/L5-horizon fast trajectory must conserve the Jacobi constant."""
    events, tol, tmax = _events_and_tol("l4_qpo")
    s0 = initial_state(20.0, 60.0, 90.0, 0.0, speed_du_per_tu(np.array([2.6]))[0])
    sol = propagate(s0, (0.0, tmax), events=events, **tol)
    js = np.array([jacobi(sol.y[:, k]) for k in range(sol.y.shape[1])])
    drift = float(np.max(np.abs(js - js[0])))
    # RK45 at rtol 1e-8 holds the Jacobi constant to ~1e-6 over ~130 days.
    assert drift < 1e-5, f"Jacobi drift {drift:.2e} too large"
