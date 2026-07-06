"""
Baseline / regression harness for the trajectory hot path.

Runs `physics.optimizer.compute_grid` over a fixed deterministic tiling for a set
of representative destinations (one per timing regime) and records the per-tile
ΔV array plus wall-clock time.  Used to:

  1. capture a scipy "golden" reference before optimising, and
  2. verify that Tier-0 / Tier-1 changes reproduce the same ΔV map.

Run:
    .venv/Scripts/python.exe -m tests.bench_trajectory --save-ref
    .venv/Scripts/python.exe -m tests.bench_trajectory --compare-ref
    .venv/Scripts/python.exe -m tests.bench_trajectory --dests earth_leo_1200 --tiles 48

The __main__ guard is required: compute_grid spawns a ProcessPoolExecutor and
Windows re-imports this module in every worker.
"""

import argparse
import os
import sys
import time

import numpy as np

# Representative destinations, one per (T_MAX, max_step, rtol) regime.
REGIME_DESTS = ["earth_leo_1200", "l1_halo", "l4_qpo"]

REF_DIR = os.path.join(os.path.dirname(__file__), "_bench_ref")


def _all_destinations():
    # Import lazily so worker re-imports stay cheap and Dash is never pulled in.
    from destinations.earth_leo   import ALL_DESTINATIONS as d1
    from destinations.l1_halo     import ALL_DESTINATIONS as d2
    from destinations.l1_lyapunov import ALL_DESTINATIONS as d3
    from destinations.l1_combined import ALL_DESTINATIONS as d4
    from destinations.l2_halo     import ALL_DESTINATIONS as d5
    from destinations.l2_lyapunov import ALL_DESTINATIONS as d6
    from destinations.l2_combined import ALL_DESTINATIONS as d7
    from destinations.l4_qpo      import ALL_DESTINATIONS as d8
    from destinations.l5_qpo      import ALL_DESTINATIONS as d9
    return {**d1, **d2, **d3, **d4, **d5, **d6, **d7, **d8, **d9}


def run_dest(dest, sites):
    from physics.optimizer import compute_grid
    t0 = time.perf_counter()
    dv, trajs, az, el, spd, cell_trajs = compute_grid(sites, dest)
    elapsed = time.perf_counter() - t0
    return dv, az, el, spd, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=48)
    ap.add_argument("--dests", nargs="*", default=REGIME_DESTS)
    ap.add_argument("--save-ref", action="store_true")
    ap.add_argument("--compare-ref", action="store_true")
    ap.add_argument("--tol-kms", type=float, default=1e-4)
    args = ap.parse_args()

    from physics.tiling import generate_tiling
    tiling = generate_tiling(args.tiles)
    sites = tiling.sites
    print(f"Tiling: {len(sites)} sites")

    all_dests = _all_destinations()
    os.makedirs(REF_DIR, exist_ok=True)

    exit_code = 0
    for dest_id in args.dests:
        dest = all_dests[dest_id]
        dv, az, el, spd, elapsed = run_dest(dest, sites)
        finite = np.isfinite(dv)
        print(f"\n=== {dest_id} ===")
        print(f"  wall: {elapsed:.2f} s   reachable: {finite.sum()}/{len(dv)}   "
              f"dV range: {np.nanmin(np.where(finite, dv, np.nan)):.4f}-"
              f"{np.nanmax(np.where(finite, dv, np.nan)):.4f} km/s")

        ref_path = os.path.join(REF_DIR, f"{dest_id}_n{args.tiles}.npz")

        if args.save_ref:
            np.savez(ref_path, dv=dv, az=az, el=el, spd=spd, elapsed=elapsed)
            print(f"  saved reference → {ref_path}")

        if args.compare_ref:
            if not os.path.exists(ref_path):
                print(f"  NO REFERENCE at {ref_path} — run --save-ref first")
                exit_code = 1
                continue
            ref = np.load(ref_path)
            rdv = ref["dv"]
            # Reachability must match exactly.
            reach_mismatch = np.isfinite(dv) != np.isfinite(rdv)
            n_reach_mismatch = int(reach_mismatch.sum())
            both = np.isfinite(dv) & np.isfinite(rdv)
            max_ddv = float(np.max(np.abs(dv[both] - rdv[both]))) if both.any() else 0.0
            speedup = ref["elapsed"] / elapsed if elapsed > 0 else float("nan")
            status = "OK"
            if n_reach_mismatch > 0 or max_ddv > args.tol_kms:
                status = "FAIL"
                exit_code = 1
            print(f"  compare: {status}  reach_mismatch={n_reach_mismatch}  "
                  f"max|Δdv|={max_ddv:.2e} km/s  speedup={speedup:.2f}×")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
