# Architecture — Lunar Mass Driver Orbital Simulation

## Overview

A Plotly Dash web application that computes and visualises the orbital transfer ΔV from every point on the Moon's surface to a selectable destination orbit. The physics engine uses the Circular Restricted 3-Body Problem (CR3BP) to model Earth–Moon dynamics. Results are cached as `.npz` files so reloads are instant.

```text
Browser ←—— Dash (app.py) ——→ background thread
                                     │
                           physics/optimizer.py
                           ProcessPoolExecutor (all CPU cores)
                                     │
                           physics/cr3bp.py  (RK45 integrator)
                           physics/coordinates.py
                                     │
                           destinations/earth_leo.py
                           (implements Destination ABC)
```

---

## Directory Layout

```text
app.py                        Dash entry point, layout, callbacks
physics/
    cr3bp.py                  CR3BP EOM, RK45 propagator, event factories
    coordinates.py            Moon surface → CR3BP rotating frame transforms
    optimizer.py              Per-site ΔV sweep, parallel grid computation
destinations/
    base.py                   Destination abstract base class
    earth_leo.py              1 200 km LEO destination (Phase 1)
visualization/
    moon_map.py               Equirectangular heatmap + Moon photo background
    trajectories.py           3-D rotating-frame trajectory viewer
assets/
    moon_surface.jpg          Background for the 2-D map panel
    earth.jpg                 (loaded but not yet applied to sphere texture)
cache/                        Auto-created; stores <dest_id>.npz result files
requirements.txt
```

---

## Non-Dimensional Unit System

All physics calculations use non-dimensional (ND) units to keep the integrator well-conditioned.

| Quantity | Symbol | Value |
|---|---|---|
| Length unit | DU | 384 400 km (Earth–Moon distance) |
| Time unit | TU | 375 190 s ≈ 4.34 days (1/mean-motion) |
| Velocity unit | VU | DU/TU ≈ 1.0245 km/s |
| Mass ratio | μ | 0.012150584 (M_Moon / M_total) |

Positions converted to km only at visualisation time (`* DU_KM`). ΔV returned to the UI in km/s.

---

## CR3BP Rotating Frame (`physics/cr3bp.py`)

The frame co-rotates with the Earth–Moon system at the mean motion:

- **Origin**: Earth–Moon barycentre
- **+x**: Earth → Moon
- **+y**: direction of Moon's orbital velocity (prograde)
- **+z**: orbit-normal (ecliptic north)
- **Earth position**: (−μ, 0, 0)
- **Moon position**: (1−μ, 0, 0)

### Equations of Motion

The standard CR3BP EOM including the Coriolis and centrifugal pseudo-forces:

```text
ẍ = 2ẏ + x − (1−μ)(x+μ)/r₁³ − μ(x−1+μ)/r₂³
ÿ = −2ẋ + y − (1−μ)y/r₁³ − μy/r₂³
z̈ = −(1−μ)z/r₁³ − μz/r₂³
```

### Integrator

`propagate()` calls `scipy.integrate.solve_ivp` with `method="RK45"`, `rtol=1e-9`, `atol=1e-11`, `max_step=0.05 TU` (~45 min). Each trajectory is propagated for up to `T_MAX_TU = 8 TU` (~35 days) or until a terminal event fires.

### Stopping Conditions

Integration halts at the **first** of these four terminal events:

| Event | Fires when | Outcome |
|---|---|---|
| `make_target_altitude_event()` | r_Earth crosses R_target inbound | **Valid** — ΔV computed at this point |
| `make_moon_impact_event()` | r_Moon ≤ 1 737.4 km | **Miss** — ΔV = ∞ |
| `make_earth_impact_event()` | r_Earth ≤ 6 371 km | **Miss** — ΔV = ∞ |
| `make_escape_event(r_max=3.5)` | r_bary > 3.5 DU (≈ 1 345 000 km) | **Miss** — craft left the system |

If none of the four events fire within 35 days the integrator also stops and the trajectory is marked unreachable (ΔV = ∞).

`make_earth_periapsis_event()` is a separate utility event not used in the grid sweep; it detects periapsis (ṙ_Earth = 0) for one-off analysis.

### Acceptance Criterion

A trajectory is valid **only** if it crosses the shell at r_Earth = R_target = **7 571 km** while moving inbound. There is no tolerance band. `solve_ivp` uses dense output interpolation to locate the crossing precisely, so the state returned is at the exact crossing point, not at the nearest integration step.

---

## Coordinate Transforms (`physics/coordinates.py`)

The Moon is synchronously locked, so its body frame co-rotates exactly with the CR3BP frame. No time-dependent rotation is needed.

**Selenographic → CR3BP alignment:**

```text
X_sel (→ Earth, lon=0)  =  −X_rot
Y_sel (east, lon=90°)   =  +Y_rot
Z_sel (north)           =  +Z_rot
```

Key functions:

- `surface_position(lat, lon)` — Moon-surface point in CR3BP DU
- `launch_velocity_rotating(lat, lon, azimuth, elevation, speed)` — velocity vector in rotating frame; no Ω×r correction because the surface is stationary in this frame
- `initial_state(...)` — concatenates position + velocity into a 6-element state vector
- `rotating_to_earth_inertial(state)` — applies `v_inertial = v_rot + Ω × r_Earth` (Ω = ẑ) at t = 0 for ΔV calculations

---

## Destination Plugin Pattern (`destinations/`)

`Destination` (in `base.py`) is an abstract base class with one required method:

```python
def compute_deltav(self, state0) -> (dv_nondim: float, trajectory: dict)
```

Returns `np.inf` if the destination is unreachable from that state. The trajectory dict has keys `t, x, y, z` (arrays) and `burns` (list of `{x, y, z, dv_kms}`).

Adding a new destination (L1 halo, DRO, etc.) means subclassing `Destination`, implementing `compute_deltav`, and registering in `ALL_DESTINATIONS`.

### Current Destination: `EarthLEO1200`

Target: 1 200 km circular orbit in the Moon's orbital plane (i = 0°).

1. Propagate with RK45. The integrator stops as soon as the trajectory crosses r = 7 571 km from Earth while moving inbound (see Acceptance Criterion above). If the trajectory reaches periapsis above 7 571 km and turns back outward without ever crossing that shell, it is a miss (ΔV = ∞).
2. At the exact crossing point, compute the circularisation ΔV in the Earth-centred inertial frame (rotating frame velocity corrected by Ω × r):

```text
ΔV = |v_inertial − v_target|

v_target = v_circ × φ̂    (tangential unit vector, prograde or retrograde)
v_circ   = √(μ_Earth / r) ≈ 7.26 km/s at 1 200 km
```

Out-of-plane velocity (vz) is fully included in the ΔV cost. The insertion mode (prograde / retrograde / both) is configurable; "both" takes the cheaper of the two. The optimiser naturally drives periapsis toward the target altitude, which minimises ΔV by making the radial component near zero at the crossing.

---

## Optimiser (`physics/optimizer.py`)

For each surface site `(lat, lon)` the optimiser performs a brute-force parameter sweep:

| Parameter | Values | Count |
|---|---|---|
| Azimuth (°, CW from North) | 0, 45, 90, 135, 180, 225, 270, 315 | 8 |
| Elevation (° above horizontal) | 0, 5 | 2 |
| Launch speed (km/s) | 2.2, 2.4, 2.5, 2.55, 2.59, 2.63, 2.68, 2.72, 2.78, 2.9 | 10 |
| **Total propagations per site** | | **160** |

The speed grid is denser around 2.55–2.65 km/s, which is the critical window for Moon → LEO transfers.

### Parallelism

`compute_grid()` uses `ProcessPoolExecutor` (all logical CPU cores). Each site is a fully independent task dispatched via `pool.submit(_site_worker, ...)`. The module-level `_site_worker` function is used (not a lambda) so it is picklable by the multiprocessing system.

After completion, `_pick_representative()` selects one best-ΔV trajectory per 45° longitude wedge (8 wedges total) for the 3-D trajectory view.

---

## Web Application (`app.py`)

Built with **Dash 4.3** + **dash-bootstrap-components** (DARKLY theme).

### Layout

```text
Header: title | destination dropdown
Progress bar + status text
Left panel:  moon_map  (equirectangular heatmap)   [dcc.Graph]
Right panel: traj_view (3-D rotating frame)         [dcc.Graph]
Controls:    max-elevation slider | param info text
Hidden:      dcc.Interval (500 ms poll), dcc.Store (active-dest)
```

### Computation Model

Dash's server runs in a single process; heavy computation must not block callbacks. The approach:

1. `on_destination_select` callback: spawns a **daemon thread** calling `_run_computation(dest_id)`.
2. `_run_computation` checks the `.npz` cache first; on miss, calls `compute_grid()` (which itself spawns worker processes).
3. Progress is written to `_compute_state` dict protected by a `threading.Lock`.
4. A `dcc.Interval` fires every 500 ms and calls `poll_progress`, which reads `_compute_state` and updates the figures and progress bar.
5. When done, the interval is disabled (`disabled=True`).

### Caching

Results are stored as `cache/<dest_id>.npz` containing:
- `dv_grid`: 2-D float array `(len(lats), len(lons))`, ΔV in km/s
- `trajs`: object array of trajectory dicts (pickled via `allow_pickle=True`)

A corrupted cache file is silently discarded and recomputed.

---

## Visualization

### Moon Map (`visualization/moon_map.py`)

- `go.Heatmap` overlay (65% opacity) on an equirectangular projection
- Background: `assets/moon_surface.jpg` loaded as base64 and added as a `layout_image`
- Colorscale: green (low ΔV) → yellow → red (high ΔV)
- `NaN` values (unreachable sites) render transparent

### 3-D Trajectory View (`visualization/trajectories.py`)

- `go.Surface` spheres for Earth and Moon, scaled 3× for visibility
- `go.Scatter3d` lines per trajectory in the CR3BP rotating frame
- Burn-point dots coloured green (low ΔV) → red (high ΔV)
- All coordinates converted to km for display; aspect ratio 3:0.9:0.9 to fit the Earth–Moon baseline

---

## Key Physical Parameters

| Parameter | Value |
|---|---|
| Earth–Moon distance (DU) | 384 400 km |
| Moon radius | 1 737.4 km |
| Earth radius | 6 371.0 km |
| Target LEO altitude | 1 200 km (r = 7 571 km) |
| Target circular speed | ≈ 7.26 km/s |
| Optimal launch speed | ≈ 2.59 km/s retrograde (az = 270°, el = 0–5°) |
| Best ΔV (near-side equatorial) | 3.05–3.4 km/s |
| Max simulation time | 8 TU ≈ 35 days |

---

## Future Extension Points

- **New destinations**: subclass `Destination`, implement `compute_deltav`, add to `ALL_DESTINATIONS` registry
- **Higher resolution grid**: reduce `GRID_LAT_STEP`/`GRID_LON_STEP` in `app.py` (20° → 162 sites, ~20 min)
- **Finer azimuth/elevation sweep**: increase `N_AZ` or `ELEVATIONS` in `optimizer.py`
- **Invariant manifold targeting**: required for L1/L2 halo or Lyapunov orbits
- **3-D Moon globe**: replace equirectangular map with a textured sphere (Three.js or dash-vtk)
