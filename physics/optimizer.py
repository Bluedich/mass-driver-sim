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
→ 160 × 5 ms ≈ 0.8 s per site, parallelised across all CPU cores via
  ProcessPoolExecutor (one process per site).
"""

import os
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

from .coordinates import initial_state, speed_du_per_tu
from .cr3bp import VU_KMS

# Azimuth sweep (degrees, CW from North) — 8 steps every 45°
N_AZ = 8
AZIMUTHS = np.linspace(0, 360, N_AZ, endpoint=False)

# Elevation angles (degrees above horizontal)
ELEVATIONS = np.array([0.0, 5.0])

# Candidate launch speeds (km/s).
# The critical window for Moon→1200 km LEO transfers is ~2.55–2.65 km/s;
# include finer sampling there plus broader range for robustness.
SPEEDS_KMS = np.array([2.2, 2.4, 2.5, 2.55, 2.59, 2.63, 2.68, 2.72, 2.78, 2.9])
SPEEDS_DU  = speed_du_per_tu(SPEEDS_KMS)   # DU/TU


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


def _site_worker(args):
    """Module-level worker so it can be pickled for ProcessPoolExecutor."""
    idx, lat, lon, destination = args
    t0 = time.perf_counter()
    dv, params = compute_site_deltav(lat, lon, destination)
    elapsed = time.perf_counter() - t0
    return idx, dv, params, elapsed


def compute_grid(lats, lons, destination, progress_cb=None, n_workers=None):
    """
    Compute suitability grid over all (lat, lon) pairs.

    Uses ProcessPoolExecutor — one process per site, up to all available CPU
    cores.  Each process runs compute_site_deltav() independently with no
    shared state, so the computation scales linearly with core count.

    Parameters
    ----------
    lats, lons : 1-D arrays (degrees)
    destination : Destination instance (must be picklable)
    progress_cb : callable(fraction 0–1) or None
    n_workers   : int or None.  None → os.cpu_count() (all logical cores).

    Returns
    -------
    dv_grid : 2-D array shape (len(lats), len(lons)), ΔV in km/s
    best_trajectories : list of representative trajectory dicts
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    pairs = [(lat, lon) for lat in lats for lon in lons]
    n     = len(pairs)

    logger.info(
        "Starting grid: %d sites, %d workers, %d propagations per site (%d total)",
        n, n_workers,
        len(ELEVATIONS) * len(AZIMUTHS) * len(SPEEDS_DU),
        n * len(ELEVATIONS) * len(AZIMUTHS) * len(SPEEDS_DU),
    )

    dv_flat     = [np.inf] * n
    params_flat = [None]   * n
    completed   = [0]
    t0_grid     = time.perf_counter()

    tasks = [(i, lat, lon, destination) for i, (lat, lon) in enumerate(pairs)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_site_worker, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            idx, dv, params, site_elapsed = fut.result()
            lat, lon = pairs[idx]
            dv_flat[idx]     = dv
            params_flat[idx] = params
            completed[0]    += 1
            dv_str = f"{dv:.3f} km/s" if np.isfinite(dv) else "unreachable"
            logger.info(
                "[%3d/%d] (%+5.1f°, %+6.1f°) → ΔV = %-16s  site: %.2f s  |  elapsed: %.1f s",
                completed[0], n, lat, lon, dv_str, site_elapsed,
                time.perf_counter() - t0_grid,
            )
            if progress_cb:
                progress_cb(completed[0] / n)

    total_elapsed = time.perf_counter() - t0_grid
    finite_dvs = [v for v in dv_flat if np.isfinite(v)]
    logger.info(
        "Grid done in %.1f s — %d/%d sites reachable, ΔV range: %.3f – %.3f km/s",
        total_elapsed, len(finite_dvs), n,
        min(finite_dvs) if finite_dvs else float("nan"),
        max(finite_dvs) if finite_dvs else float("nan"),
    )

    dv_grid = np.array(dv_flat, dtype=float).reshape(len(lats), len(lons))
    trajs   = _pick_representative(lats, lons, dv_grid, params_flat, pairs)

    return dv_grid, trajs


def _pick_representative(lats, lons, dv_grid, params_flat, pairs, n_wedges=8):
    """
    Select one trajectory per longitude wedge for the 3-D visualisation.
    Returns a list of trajectory dicts (valid only).
    """
    wedge_size = 360.0 / n_wedges
    best = {}   # wedge_idx → (dv, params)

    for i, (lat, lon) in enumerate(pairs):
        dv = dv_grid.flat[i]
        if not np.isfinite(dv):
            continue
        widx = int((lon % 360) / wedge_size) % n_wedges
        if widx not in best or dv < best[widx][0]:
            best[widx] = (dv, params_flat[i])

    trajs = []
    for widx in range(n_wedges):
        if widx in best and best[widx][1] is not None:
            p = best[widx][1]
            if p.get("trajectory") is not None:
                trajs.append(p["trajectory"])
    return trajs
