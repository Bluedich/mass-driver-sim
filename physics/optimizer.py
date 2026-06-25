"""
Per-site ΔV optimiser — fast sweep variant.

For each (lat, lon) on the Moon, sweep over azimuth and a small set of
candidate launch speeds, returning the minimum total post-launch ΔV.

Mass-driver energy is NOT counted; only on-board burns after launch matter.

Performance design
------------------
- 8 azimuth steps (every 45°)    ×
- 2 elevation angles (0°, 5°)    ×
- 10 candidate speeds             = 160 propagations per site
- Each propagation: ~5 ms for a fast (low-tol) integrator
→ 160 × 5 ms ≈ 0.8 s per site serial; parallelised at propagation level
  (N_sites × 160 tasks) via ProcessPoolExecutor with a pool initializer so
  the destination object is pickled once per worker, not once per task.
"""

import os
import logging
import time
import itertools
import numpy as np

logger = logging.getLogger(__name__)

from .coordinates import initial_state, speed_du_per_tu
from .cr3bp import VU_KMS

# Azimuth sweep (degrees, CW from North) — 8 steps every 45°
N_AZ = 8
AZIMUTHS = np.linspace(0, 360, N_AZ, endpoint=False)

MAX_N_AZ      = 1440
MAX_AZIMUTHS  = np.linspace(0, 360, MAX_N_AZ, endpoint=False)

# Elevation angles (degrees above horizontal)
ELEVATIONS = np.array([0.0, 5.0])

# Candidate launch speeds (km/s).
# The critical window for Moon→1200 km LEO transfers is ~2.55–2.65 km/s;
# include finer sampling there plus broader range for robustness.
SPEEDS_KMS = np.array([2.2, 2.4, 2.5, 2.55, 2.59, 2.63, 2.68, 2.72, 2.78, 2.9])
SPEEDS_DU  = speed_du_per_tu(SPEEDS_KMS)   # DU/TU

MAX_SPEEDS_KMS = np.linspace(2.2, 2.9, 1000)
MAX_SPEEDS_DU  = speed_du_per_tu(MAX_SPEEDS_KMS)

N_PROPS = len(ELEVATIONS) * N_AZ * len(SPEEDS_KMS)   # propagations per site (160)


def compute_site_deltav(lat_deg, lon_deg, destination):
    """
    Minimum post-launch ΔV (km/s) for a surface site.

    Returns
    -------
    best_dv_kms : float  (np.inf if no valid trajectory found)
    best_params : dict or None
        Keys: azimuth_deg, elevation_deg, speed_kms, trajectory
    """
    best_dv_kms = np.inf
    best_params = None
    n_propagations = 0
    n_hits = 0

    for el in ELEVATIONS:
        for az in AZIMUTHS:
            for v_du in SPEEDS_DU:
                t_prop = time.perf_counter()
                state0 = initial_state(lat_deg, lon_deg, az, el, v_du)
                dv_nd, traj = destination.compute_deltav(state0)
                n_propagations += 1

                logger.debug(
                    "  prop az=%3.0f° el=%.0f° v=%.3f km/s → dv=%s  (%.1f ms)",
                    az, el, v_du * VU_KMS,
                    f"{dv_nd * VU_KMS:.3f} km/s" if np.isfinite(dv_nd) else "∞",
                    (time.perf_counter() - t_prop) * 1e3,
                )

                if not np.isfinite(dv_nd):
                    continue

                n_hits += 1
                dv_kms = dv_nd * VU_KMS
                if dv_kms < best_dv_kms:
                    best_dv_kms = dv_kms
                    best_params = {
                        "azimuth_deg":   az,
                        "elevation_deg": el,
                        "speed_kms":     v_du * VU_KMS,
                        "trajectory":    traj,
                    }

    logger.debug(
        "  site (%+.0f°, %+.0f°): %d/%d hits, best ΔV=%s",
        lat_deg, lon_deg, n_hits, n_propagations,
        f"{best_dv_kms:.3f} km/s" if np.isfinite(best_dv_kms) else "∞",
    )
    return best_dv_kms, best_params


# ── Process-pool plumbing ─────────────────────────────────────────────────────
# The destination is passed to each worker process once via the pool
# initializer, avoiding repeated pickling for every propagation task.

_DEST = None   # set in each worker process by _pool_init


def _pool_init(destination):
    global _DEST
    _DEST = destination
    # Prevent OpenBLAS/MKL from spawning per-worker thread teams; each worker
    # runs one propagation at a time so extra threads only waste memory.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _prop_worker(args):
    """One task = one propagation. Uses process-global _DEST (set by _pool_init)."""
    site_idx, lat, lon, el, az, v_du = args
    t0     = time.perf_counter()
    state0 = initial_state(lat, lon, az, el, v_du)
    dv_nd, traj = _DEST.compute_deltav(state0)
    elapsed = time.perf_counter() - t0

    logger.debug(
        "  prop (%+.0f°,%+.0f°) az=%3.0f° el=%.0f° v=%.3f km/s → %s  (%.1f ms)",
        lat, lon, az, el, v_du * VU_KMS,
        f"{dv_nd * VU_KMS:.3f} km/s" if np.isfinite(dv_nd) else "∞",
        elapsed * 1e3,
    )
    # Only the parent's best-per-site trajectory is ever read, and only when
    # dv_nd is finite — null the trajectory for misses so we don't pickle a
    # large array back from every unreachable propagation.
    return site_idx, el, az, v_du, dv_nd, (traj if np.isfinite(dv_nd) else None), elapsed


def compute_grid(lats, lons, destination, progress_cb=None, site_cb=None, n_workers=None,
                 azimuths=None, elevations=None, speeds_kms=None):
    """
    Compute suitability grid over all (lat, lon) pairs.

    Submits one task per propagation (N_sites × n_props) to a ProcessPoolExecutor.
    The destination is passed once per worker process via the pool initializer,
    so pickling overhead is O(n_workers), not O(n_tasks).  Workers pull tasks
    from a shared queue, eliminating idle time from per-site load imbalance.

    Parameters
    ----------
    lats, lons : 1-D arrays (degrees)
    destination : Destination instance (must be picklable)
    progress_cb : callable(props_done: int, props_total: int) or None
        Called after every completed propagation.
    site_cb     : callable(sites_done: int, sites_total: int) or None
        Called when a full site's n_props propagations are all back.
    n_workers   : int or None.  None → os.cpu_count() (all logical cores).
    azimuths    : 1-D array of azimuth angles (°, CW from North), or None → default
    elevations  : 1-D array of elevation angles (° above horizontal), or None → default
    speeds_kms  : 1-D array of launch speeds (km/s), or None → default

    Returns
    -------
    dv_grid : 2-D array shape (len(lats), len(lons)), ΔV in km/s
    best_trajectories : list of representative trajectory dicts
    """
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

    azimuths_use   = AZIMUTHS   if azimuths   is None else np.asarray(azimuths,   dtype=float)
    elevations_use = ELEVATIONS if elevations  is None else np.asarray(elevations, dtype=float)
    speeds_kms_use = SPEEDS_KMS if speeds_kms  is None else np.asarray(speeds_kms, dtype=float)
    speeds_du_use  = speed_du_per_tu(speeds_kms_use)
    n_props        = len(elevations_use) * len(azimuths_use) * len(speeds_kms_use)

    if n_workers is None:
        # Use at most half the logical cores; spawning too many scipy/OpenBLAS
        # processes exhausts memory before the work is done.
        n_workers = max(1, (os.cpu_count() or 4) // 2)

    pairs = [(lat, lon) for lat in lats for lon in lons]
    n     = len(pairs)

    # Polar deduplication: lat = ±90° is a single physical point regardless of
    # longitude.  For each polar latitude, compute propagations at only the lon
    # closest to 0° (the sub-Earth reference) and broadcast the result to all
    # other lons in that row.  This avoids redundant work while keeping the
    # rectangular grid shape that the heatmap expects.
    polar_primary = {}  # lat_value -> site_idx of the primary (computed) site
    for site_idx, (lat, lon) in enumerate(pairs):
        if abs(lat) == 90.0:
            if lat not in polar_primary or abs(lon) < abs(pairs[polar_primary[lat]][1]):
                polar_primary[lat] = site_idx

    def _is_compute(site_idx, lat):
        if abs(lat) < 90.0:
            return True
        return site_idx == polar_primary.get(lat)

    n_compute   = sum(_is_compute(i, lat) for i, (lat, _) in enumerate(pairs))
    total_props = n_compute * n_props

    logger.info(
        "Starting grid: %d cells (%d compute sites), %d workers, %d props/site (%d total)",
        n, n_compute, n_workers, n_props, total_props,
    )

    # Per-site accumulators
    site_best    = {i: (np.inf, None) for i in range(n)}  # (dv_kms, params)
    site_counts  = {i: 0 for i in range(n)}               # propagations returned
    props_done   = [0]
    sites_done   = [0]
    t0_grid      = time.perf_counter()

    # Lazy task stream — one task per propagation, never materialised as a list.
    # For large sweeps (e.g. 1440 az × 1000 speeds × N sites) the full list would
    # be tens of millions of tuples, and submitting them all at once would create
    # an equal number of Future/_WorkItem objects in the parent before any result
    # is consumed — exhausting memory and stalling the run.
    def _task_iter():
        for site_idx, (lat, lon) in enumerate(pairs):
            if not _is_compute(site_idx, lat):
                continue
            for el in elevations_use:
                for az in azimuths_use:
                    for v_du in speeds_du_use:
                        yield (site_idx, lat, lon, el, az, v_du)

    def _handle_result(fut):
        site_idx, el, az, v_du, dv_nd, traj, prop_elapsed = fut.result()
        lat, lon = pairs[site_idx]

        if np.isfinite(dv_nd):
            dv_kms = dv_nd * VU_KMS
            if dv_kms < site_best[site_idx][0]:
                site_best[site_idx] = (dv_kms, {
                    "azimuth_deg":   az,
                    "elevation_deg": el,
                    "speed_kms":     v_du * VU_KMS,
                    "trajectory":    traj,
                })

        props_done[0] += 1
        if progress_cb:
            progress_cb(props_done[0], total_props)

        site_counts[site_idx] += 1
        if site_counts[site_idx] == n_props:
            sites_done[0] += 1
            dv, params = site_best[site_idx]
            dv_str = f"{dv:.3f} km/s" if np.isfinite(dv) else "unreachable"
            logger.info(
                "[%3d/%d] (%+5.1f°, %+6.1f°) → ΔV = %-16s  elapsed: %.1f s",
                sites_done[0], n_compute, lat, lon, dv_str,
                time.perf_counter() - t0_grid,
            )
            if site_cb:
                site_cb(sites_done[0], n_compute)

    # Bounded sliding window: keep at most ~2× n_workers futures in flight, so
    # parent memory stays O(n_workers) regardless of total task count.  Seed the
    # window, then submit one new task for each completed one until drained.
    max_inflight = 2 * n_workers

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_pool_init,
        initargs=(destination,),
    ) as pool:
        task_gen = _task_iter()
        inflight = {pool.submit(_prop_worker, task)
                    for task in itertools.islice(task_gen, max_inflight)}

        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                _handle_result(fut)
                nxt = next(task_gen, None)
                if nxt is not None:
                    inflight.add(pool.submit(_prop_worker, nxt))

    # Broadcast polar primary results to all secondary polar sites.
    # Secondary sites get the same ΔV (for heatmap colour) but params=None
    # (no arrow, no trajectory) so only the primary site shows an arrow.
    for plat, primary_idx in polar_primary.items():
        dv_pole, _ = site_best[primary_idx]
        for sec_idx, (slat, _) in enumerate(pairs):
            if slat == plat and sec_idx != primary_idx:
                site_best[sec_idx] = (dv_pole, None)

    total_elapsed = time.perf_counter() - t0_grid
    dv_flat     = [site_best[i][0] for i in range(n)]
    params_flat = [site_best[i][1] for i in range(n)]
    finite_dvs  = [v for v in dv_flat if np.isfinite(v)]
    logger.info(
        "Grid done in %.1f s — %d/%d sites reachable, ΔV range: %.3f – %.3f km/s",
        total_elapsed, len(finite_dvs), n,
        min(finite_dvs) if finite_dvs else float("nan"),
        max(finite_dvs) if finite_dvs else float("nan"),
    )

    dv_grid  = np.array(dv_flat, dtype=float).reshape(len(lats), len(lons))
    az_flat  = [p["azimuth_deg"]   if p else np.nan for p in params_flat]
    el_flat  = [p["elevation_deg"] if p else np.nan for p in params_flat]
    spd_flat = [p["speed_kms"]     if p else np.nan for p in params_flat]
    az_grid  = np.array(az_flat,  dtype=float).reshape(len(lats), len(lons))
    el_grid  = np.array(el_flat,  dtype=float).reshape(len(lats), len(lons))
    spd_grid = np.array(spd_flat, dtype=float).reshape(len(lats), len(lons))
    trajs    = _pick_representative(lats, lons, dv_grid, params_flat, pairs)

    cell_trajs = np.empty((len(lats), len(lons)), dtype=object)
    for i_site, p in enumerate(params_flat):
        ri, rj = divmod(i_site, len(lons))
        cell_trajs[ri, rj] = p["trajectory"] if p is not None else None

    return dv_grid, trajs, az_grid, el_grid, spd_grid, cell_trajs


def _pick_representative(lats, lons, dv_grid, params_flat, pairs, n=10):
    """
    Select the n sites with the lowest ΔV for the 3-D visualisation.
    Returns a list of trajectory dicts (valid only).
    """
    ranked = sorted(
        ((dv_grid.flat[i], params_flat[i]) for i, _ in enumerate(pairs)
         if np.isfinite(dv_grid.flat[i]) and params_flat[i] is not None),
        key=lambda t: t[0],
    )

    trajs = []
    for _, p in ranked:
        if p.get("trajectory") is not None:
            trajs.append(p["trajectory"])
        if len(trajs) == n:
            break
    return trajs
