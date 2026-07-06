"""
Lunar Mass Driver — Dash application module.

Imported by app.py (the entry point). Lives in its own module so that
ProcessPoolExecutor worker processes, which re-execute the __main__ script
(app.py) on Windows, never import this file and never pay the cost of
initialising Dash/Plotly.
"""

import os
import re
import json
import logging
import threading
import time
from datetime import datetime
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ALL, ctx

from destinations.earth_leo   import ALL_DESTINATIONS as _D_LEO
from destinations.l1_halo     import ALL_DESTINATIONS as _D_L1
from destinations.l1_lyapunov import ALL_DESTINATIONS as _D_LYAP
from destinations.l1_combined import ALL_DESTINATIONS as _D_L1C
from destinations.l2_halo     import ALL_DESTINATIONS as _D_L2
from destinations.l2_lyapunov import ALL_DESTINATIONS as _D_L2LYAP
from destinations.l2_combined import ALL_DESTINATIONS as _D_L2C
ALL_DESTINATIONS = {**_D_LEO, **_D_L1, **_D_LYAP, **_D_L1C,
                    **_D_L2, **_D_L2LYAP, **_D_L2C}
from visualization.moon_map import build_moon_map, build_empty_moon_map
from visualization.trajectories import build_trajectory_view, build_empty_trajectory_view, scene_bounds, fixed_scene_bounds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lunar_sim")


def _selected_orbit_ids(trajs):
    """Orbit ids the given trajectories' burns insert into."""
    return {b["orbit_id"] for t in trajs for b in t.get("burns", [])
            if "orbit_id" in b}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_ARCH_MD_PATH = os.path.join(os.path.dirname(__file__), "ARCHITECTURE.md")
try:
    with open(_ARCH_MD_PATH, encoding="utf-8") as _f:
        _ARCH_MD = _f.read()
except FileNotFoundError:
    _ARCH_MD = "*ARCHITECTURE.md not found.*"

# ── Grid resolution ──────────────────────────────────────────────────────────
GRID_LAT_STEP = 30
GRID_LON_STEP = 30

# Poles are included as single representative sites (all lons are the same
# physical point at ±90°, so only one propagation set is run per pole).
# Longitudes span the full 360° globe at 30° steps (−180°…+180°). Both ±180°
# are included so the far-side meridian renders as a seamless half-cell on each
# map edge; they are the same physical point and yield identical ΔV.
LATS = np.array([-90, -60, -30,  0, 30, 60, 90], dtype=float)
LONS = np.arange(-180, 181, 30, dtype=float)

# ── Shared computation state ──────────────────────────────────────────────────
_compute_state = {
    "running":     False,
    "progress":    0.0,
    "props_done":  0,
    "props_total": 0,
    "sites_done":  0,
    "sites_total": 0,
    "t_start":     None,
    "result":      None,
    "dest_id":     None,
    "cache_key":   None,
    "error":       None,
}
_lock = threading.Lock()


def _make_sweep_arrays(n_az, n_el, max_el, n_sp):
    """Build azimuth, elevation, and speed arrays from UI parameter counts."""
    azimuths  = np.linspace(0, 360, int(n_az), endpoint=False)
    if int(n_el) <= 1 or float(max_el) == 0:
        elevations = np.array([0.0])
    else:
        elevations = np.linspace(0.0, float(max_el), int(n_el))
    speeds_kms = np.linspace(2.2, 2.9, int(n_sp))
    return azimuths, elevations, speeds_kms


def _format_eta(seconds):
    """Human-readable remaining-time string (e.g. '45 s', '3 min 20 s', '1 h 5 min')."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60} s"
    return f"{seconds // 3600} h {(seconds % 3600) // 60} min"


def _make_cache_key(dest_id, n_az, n_el, max_el, n_sp, insertion_mode="prograde"):
    return (f"{dest_id}_az{int(n_az)}_el{int(n_el)}x{int(max_el)}"
            f"_sp{int(n_sp)}_ins{insertion_mode}_g{len(LATS)}x{len(LONS)}")


_CACHE_KEY_RE = re.compile(
    r"^(?P<dest>.+)_az(?P<n_az>\d+)_el(?P<n_el>\d+)x(?P<max_el>\d+)"
    r"_sp(?P<n_sp>\d+)_ins(?P<ins>[a-z]+)_g(?P<n_lat>\d+)x(?P<n_lon>\d+)$"
)


def _parse_cache_key(stem):
    """Reconstruct settings from a cache-key filename stem (None if it doesn't match)."""
    m = _CACHE_KEY_RE.match(stem)
    if not m:
        return None
    dest_id = m.group("dest")
    dest = ALL_DESTINATIONS.get(dest_id)
    return {
        "dest_id": dest_id,
        "label": dest.label if dest else dest_id,
        "n_az": int(m.group("n_az")), "n_el": int(m.group("n_el")),
        "max_el": int(m.group("max_el")), "n_sp": int(m.group("n_sp")),
        "insertion_mode": m.group("ins"),
        "n_lat": int(m.group("n_lat")), "n_lon": int(m.group("n_lon")),
    }


def _load_saved_meta():
    """Scan the cache dir and return one metadata dict per saved simulation.

    Reads the sidecar JSON when present; otherwise backfills it from the filename
    (settings), file mtime (timestamp) and a one-time NPZ load (best ΔV). Files
    whose grid no longer matches the current LATS/LONS are skipped — they can't be
    reopened against the current code.
    """
    metas = []
    try:
        names = os.listdir(CACHE_DIR)
    except FileNotFoundError:
        return metas

    for name in names:
        if not name.endswith(".npz") or name == "l1_halos.npz":
            continue
        stem = name[:-4]
        npz_path = os.path.join(CACHE_DIR, name)
        json_path = os.path.join(CACHE_DIR, f"{stem}.json")

        meta = None
        if os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception as exc:
                logger.warning("Sidecar read failed for %s: %s", stem, exc)

        if meta is None:
            parsed = _parse_cache_key(stem)
            if parsed is None:
                continue
            best_dv = None
            try:
                data = np.load(npz_path, allow_pickle=True)
                dv = data["dv_grid"]
                finite = dv[np.isfinite(dv)]
                best_dv = float(np.min(finite)) if finite.size else None
            except Exception as exc:
                logger.warning("Best-ΔV backfill failed for %s: %s", stem, exc)
            meta = {**parsed, "created_at": os.path.getmtime(npz_path), "best_dv": best_dv}
            try:
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh)
            except Exception as exc:
                logger.warning("Sidecar backfill write failed for %s: %s", stem, exc)

        if meta.get("n_lat") != len(LATS) or meta.get("n_lon") != len(LONS):
            continue
        meta["cache_key"] = stem
        metas.append(meta)

    return metas


def build_saved_list():
    """Build the saved-simulations modal body: scenarios as expandable groups."""
    metas = _load_saved_meta()
    if not metas:
        return html.Div("No saved simulations yet.",
                        className="text-gray-500 text-sm py-6 text-center")

    groups = {}
    for m in metas:
        groups.setdefault(m["dest_id"], []).append(m)

    # Order groups by their most recent sim, sims within a group newest-first.
    ordered = sorted(groups.items(),
                     key=lambda kv: max(s.get("created_at", 0) for s in kv[1]),
                     reverse=True)

    sections = []
    for dest_id, sims in ordered:
        sims.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        label = sims[0].get("label", dest_id)
        rows = [_saved_row(s) for s in sims]
        sections.append(html.Details(
            open=True,
            className="border border-neutral-800 rounded-lg overflow-hidden",
            children=[
                html.Summary(
                    className=(
                        "flex items-center justify-between px-3 py-2 cursor-pointer "
                        "bg-neutral-800/60 hover:bg-neutral-800 text-gray-200 "
                        "text-sm font-medium select-none"
                    ),
                    children=[
                        html.Span(label),
                        html.Span(f"{len(sims)} saved",
                                  className="text-gray-500 text-xs font-normal"),
                    ],
                ),
                html.Div(className="divide-y divide-neutral-800", children=rows),
            ],
        ))
    return html.Div(className="flex flex-col gap-3", children=sections)


def _saved_row(meta):
    """One saved-simulation row: settings chips, timestamp, best ΔV, Open/Delete."""
    key = meta["cache_key"]
    when = ""
    try:
        when = datetime.fromtimestamp(meta["created_at"]).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    best = meta.get("best_dv")
    best_txt = f"{best:.2f} km/s" if isinstance(best, (int, float)) else "—"

    return html.Div(
        className="flex items-center justify-between gap-3 px-3 py-2",
        children=[
            html.Div(className="flex flex-wrap items-center gap-1.5", children=[
                html.Span(f"{meta['n_az']} az × {meta['n_el']}×{meta['max_el']}° el "
                          f"× {meta['n_sp']} spd", className=_CHIP_CLS),
                html.Span(meta["insertion_mode"], className=_CHIP_CLS),
                html.Span(f"best ΔV: {best_txt}", className=_CHIP_CLS),
                html.Span(when, className="text-gray-500 text-xs ml-1"),
            ]),
            html.Div(className="flex items-center gap-1.5 shrink-0", children=[
                html.Button(
                    "Open", id={"type": "sim-open", "key": key}, n_clicks=0,
                    className=(
                        "px-2.5 py-1 text-xs rounded border border-gray-600 text-gray-300 "
                        "hover:border-gray-300 hover:text-white transition-colors cursor-pointer"
                    ),
                ),
                html.Button(
                    "Delete", id={"type": "sim-delete", "key": key}, n_clicks=0,
                    className=(
                        "px-2.5 py-1 text-xs rounded border border-red-900 text-red-400 "
                        "hover:border-red-500 hover:text-red-300 transition-colors cursor-pointer"
                    ),
                ),
            ]),
        ],
    )


def _run_computation(dest_id, n_az, n_el, max_el, n_sp, insertion_mode="prograde"):
    """Background thread: compute suitability grid and sample trajectories."""
    dest = ALL_DESTINATIONS[dest_id]
    dest.insertion_mode = insertion_mode
    cache_key  = _make_cache_key(dest_id, n_az, n_el, max_el, n_sp, insertion_mode)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")

    with _lock:
        _compute_state["running"]   = True
        _compute_state["progress"]  = 0.0
        _compute_state["result"]    = None
        _compute_state["dest_id"]   = dest_id
        _compute_state["cache_key"] = cache_key
        _compute_state["error"]     = None

    n_polar_lats    = sum(1 for lat in LATS if abs(lat) == 90.0)
    n_compute_sites = (len(LATS) - n_polar_lats) * len(LONS) + n_polar_lats
    n_sites         = len(LATS) * len(LONS)   # total cells (includes secondary polar)
    logger.info("Computation requested for '%s' (%d×%d = %d cells, %d compute sites)",
                dest.label, len(LATS), len(LONS), n_sites, n_compute_sites)

    if os.path.exists(cache_path):
        logger.info("Cache hit: %s — loading from disk", cache_path)
        t0 = time.perf_counter()
        try:
            data     = np.load(cache_path, allow_pickle=True)
            dv_grid    = data["dv_grid"]
            trajs      = list(data["trajs"])
            az_grid    = data["az_grid"]    if "az_grid"    in data else None
            el_grid    = data["el_grid"]    if "el_grid"    in data else None
            spd_grid   = data["spd_grid"]   if "spd_grid"   in data else None
            cell_trajs = data["cell_trajs"] if "cell_trajs" in data else None
            logger.info("Cache loaded in %.2f s", time.perf_counter() - t0)
            with _lock:
                _compute_state["result"]   = (dv_grid, trajs, az_grid, el_grid, spd_grid, dest_id, cache_key, cell_trajs)
                _compute_state["running"]  = False
                _compute_state["progress"] = 1.0
            return
        except Exception as exc:
            logger.warning("Cache load failed (%s) — recomputing", exc)

    from physics.optimizer import compute_grid

    azimuths, elevations, speeds_kms = _make_sweep_arrays(n_az, n_el, max_el, n_sp)
    n_props_total = n_compute_sites * len(azimuths) * len(elevations) * len(speeds_kms)
    with _lock:
        _compute_state["props_done"]  = 0
        _compute_state["props_total"] = n_props_total
        _compute_state["sites_done"]  = 0
        _compute_state["sites_total"] = n_compute_sites
        _compute_state["t_start"]     = time.perf_counter()

    def _progress(props_done, props_total):
        with _lock:
            _compute_state["progress"]   = props_done / props_total
            _compute_state["props_done"] = props_done

    def _site_done(sites_done, sites_total):
        with _lock:
            _compute_state["sites_done"] = sites_done

    t0 = time.perf_counter()
    try:
        dv_grid, trajs, az_grid, el_grid, spd_grid, cell_trajs = compute_grid(
            LATS, LONS, dest,
            progress_cb=_progress,
            site_cb=_site_done,
            azimuths=azimuths,
            elevations=elevations,
            speeds_kms=speeds_kms,
        )
    except Exception as exc:
        logger.exception("Grid computation failed")
        with _lock:
            _compute_state["running"]  = False
            _compute_state["progress"] = 0.0
            _compute_state["error"]    = str(exc)
        return
    logger.info("Grid computation finished in %.1f s", time.perf_counter() - t0)

    t_save = time.perf_counter()
    try:
        np.savez(cache_path, dv_grid=dv_grid, trajs=np.array(trajs, dtype=object),
                 az_grid=az_grid, el_grid=el_grid, spd_grid=spd_grid,
                 cell_trajs=cell_trajs)
        logger.info("Cache saved to %s (%.2f s)", cache_path, time.perf_counter() - t_save)
    except Exception as exc:
        logger.warning("Cache save failed: %s", exc)

    try:
        finite = dv_grid[np.isfinite(dv_grid)]
        best_dv = float(np.min(finite)) if finite.size else None
        meta = {
            "dest_id": dest_id, "label": dest.label,
            "n_az": int(n_az), "n_el": int(n_el), "max_el": int(max_el),
            "n_sp": int(n_sp), "insertion_mode": insertion_mode,
            "n_lat": len(LATS), "n_lon": len(LONS),
            "created_at": time.time(), "best_dv": best_dv,
        }
        with open(os.path.join(CACHE_DIR, f"{cache_key}.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    except Exception as exc:
        logger.warning("Sidecar metadata save failed: %s", exc)

    with _lock:
        _compute_state["result"]   = (dv_grid, trajs, az_grid, el_grid, spd_grid, dest_id, cache_key, cell_trajs)
        _compute_state["running"]  = False
        _compute_state["progress"] = 1.0


# ── Dash app ──────────────────────────────────────────────────────────────────
_INDEX_STRING = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
    <script src="https://cdn.jsdelivr.net/npm/preline/dist/preline.js"></script>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>"""

app = dash.Dash(
    __name__,
    title="Lunar Mass Driver Sim",
    update_title=None,
    index_string=_INDEX_STRING,
)

DEST_OPTIONS = [{"label": d.label, "value": d.id} for d in ALL_DESTINATIONS.values()]

_BAR_HIDDEN    = "hidden h-1.5 bg-neutral-800 rounded-full mb-1 overflow-hidden"
_BAR_SHOWN     = "h-1.5 bg-neutral-800 rounded-full mb-1 overflow-hidden"
_MODAL_CLOSED  = "hidden fixed inset-0 z-50 bg-black/70 items-center justify-center"
_MODAL_OPEN    = "flex fixed inset-0 z-50 bg-black/70 items-center justify-center"
_CHIP_CLS      = ("px-2 py-0.5 rounded text-xs font-mono "
                  "bg-neutral-800 text-gray-400 border border-neutral-700")

app.layout = html.Div(
    className="min-h-screen bg-[#0d0d0d] p-3 text-gray-200",
    children=[

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div(className="flex items-center justify-between mb-3", children=[
        html.Div(className="flex items-center gap-3", children=[
            html.H4("Lunar Mass Driver — Orbital Suitability",
                    className="text-gray-200 text-lg font-semibold m-0"),
            html.Button(
                "?", id="help-btn",
                className=(
                    "w-7 h-7 rounded-full border border-gray-500 text-gray-400 text-sm "
                    "hover:border-gray-300 hover:text-gray-200 flex items-center "
                    "justify-center transition-colors cursor-pointer leading-none shrink-0"
                ),
            ),
        ]),
        html.Div(className="flex items-end gap-2", children=[
            html.Div(className="w-64", children=[
                html.Label("Destination", className="text-gray-500 text-xs block mb-1"),
                dcc.Dropdown(
                    id="dest-dropdown",
                    options=DEST_OPTIONS,
                    value=None,
                    placeholder="Select destination…",
                    clearable=False,
                    style={"backgroundColor": "#1e1e1e", "color": "#eee",
                           "border": "1px solid #444"},
                ),
            ]),
            html.Button(
                "Calculate", id="calc-btn", n_clicks=0, disabled=True,
                className=(
                    "px-3 py-1.5 text-sm rounded border border-gray-600 text-gray-300 "
                    "hover:border-gray-300 hover:text-white transition-colors cursor-pointer "
                    "whitespace-nowrap disabled:opacity-40 disabled:cursor-not-allowed"
                ),
            ),
            html.Button(
                "Saved", id="saved-btn", n_clicks=0,
                className=(
                    "px-3 py-1.5 text-sm rounded border border-gray-600 text-gray-300 "
                    "hover:border-gray-300 hover:text-white transition-colors cursor-pointer "
                    "whitespace-nowrap"
                ),
            ),
        ]),
    ]),

    # ── Progress bar ──────────────────────────────────────────────────────────
    html.Div(className="mb-2", children=[
        html.Div(id="progress-wrap", className=_BAR_HIDDEN, children=[
            html.Div(id="progress-fill",
                     className="h-full bg-green-500 rounded-full transition-all duration-300",
                     style={"width": "0%"}),
        ]),
        html.Div(id="status-text", className="text-gray-500 text-xs h-4"),
    ]),

    # ── Main panels ───────────────────────────────────────────────────────────
    html.Div(className="grid grid-cols-2 gap-3", children=[
        html.Div(className="rounded-lg overflow-hidden",
                 children=dcc.Graph(id="moon-map", figure=build_empty_moon_map(),
                                    config={"displayModeBar": False})),
        html.Div(className="flex flex-col gap-1", children=[
            html.Div(className="flex gap-1", children=[
                html.Button("Center: Moon", id="center-moon-btn", n_clicks=0,
                            className=(
                                "px-2 py-0.5 text-xs rounded border border-neutral-700 "
                                "text-gray-400 hover:border-gray-500 hover:text-gray-200 "
                                "transition-colors cursor-pointer"
                            )),
                html.Button("Center: Earth", id="center-earth-btn", n_clicks=0,
                            className=(
                                "px-2 py-0.5 text-xs rounded border border-neutral-700 "
                                "text-gray-400 hover:border-gray-500 hover:text-gray-200 "
                                "transition-colors cursor-pointer"
                            )),
            ]),
            html.Div(className="rounded-lg overflow-hidden",
                     children=dcc.Graph(id="traj-view", figure=build_empty_trajectory_view(),
                                        config={"displayModeBar": True,
                                                "modeBarButtonsToRemove": ["toImage"]})),
        ]),
    ]),

    # ── Controls ──────────────────────────────────────────────────────────────
    html.Div(className="bg-neutral-900 rounded-lg px-4 py-3 mt-3", children=[
        html.Div(className="flex items-end gap-6 flex-wrap mb-3", children=[
            html.Div(className="flex flex-col gap-1", children=[
                html.Label("Azimuths", htmlFor="n-azimuths",
                           className="text-gray-400 text-xs font-medium"),
                dcc.Input(id="n-azimuths", type="number", value=8, min=2, max=1440, step=2,
                          debounce=True,
                          className=(
                              "w-20 bg-neutral-800 border border-neutral-700 rounded "
                              "text-gray-200 text-sm px-2 py-1 focus:outline-none "
                              "focus:border-gray-500"
                          )),
            ]),
            html.Div(className="flex flex-col gap-1", children=[
                html.Label("Max elevation (°)", htmlFor="max-elevation",
                           className="text-gray-400 text-xs font-medium"),
                dcc.Input(id="max-elevation", type="number", value=5, min=0, max=90, step=5,
                          debounce=True,
                          className=(
                              "w-20 bg-neutral-800 border border-neutral-700 rounded "
                              "text-gray-200 text-sm px-2 py-1 focus:outline-none "
                              "focus:border-gray-500"
                          )),
            ]),
            html.Div(className="flex flex-col gap-1", children=[
                html.Label("Elevation steps", htmlFor="n-elevations",
                           className="text-gray-400 text-xs font-medium"),
                dcc.Input(id="n-elevations", type="number", value=2, min=1, max=20, step=1,
                          debounce=True,
                          className=(
                              "w-20 bg-neutral-800 border border-neutral-700 rounded "
                              "text-gray-200 text-sm px-2 py-1 focus:outline-none "
                              "focus:border-gray-500"
                          )),
            ]),
            html.Div(className="flex flex-col gap-1", children=[
                html.Label("Speed candidates", htmlFor="n-speeds",
                           className="text-gray-400 text-xs font-medium"),
                dcc.Input(id="n-speeds", type="number", value=10, min=3, max=1000, step=1,
                          debounce=True,
                          className=(
                              "w-20 bg-neutral-800 border border-neutral-700 rounded "
                              "text-gray-200 text-sm px-2 py-1 focus:outline-none "
                              "focus:border-gray-500"
                          )),
            ]),
            html.Div(className="flex flex-col gap-1", children=[
                html.Label("Insertion", htmlFor="insertion-mode",
                           className="text-gray-400 text-xs font-medium"),
                dcc.Dropdown(
                    id="insertion-mode",
                    options=[
                        {"label": "Prograde only",   "value": "prograde"},
                        {"label": "Retrograde only", "value": "retrograde"},
                        {"label": "Both (cheapest)", "value": "both"},
                    ],
                    value="prograde",
                    clearable=False,
                    style={"backgroundColor": "#262626", "color": "#e5e5e5",
                           "border": "1px solid #525252", "minWidth": "150px"},
                ),
            ]),
        ]),
        html.Div(className="flex flex-wrap items-center gap-2", id="param-info"),
    ]),

    # ── Hidden state ──────────────────────────────────────────────────────────
    dcc.Interval(id="poll-interval", interval=500, n_intervals=0, disabled=True),
    dcc.Store(id="active-dest", data=None),
    dcc.Store(id="selected-cell", data=None),

    # ── Help modal ────────────────────────────────────────────────────────────
    html.Div(
        id="help-modal",
        className=_MODAL_CLOSED,
        children=html.Div(
            className=(
                "bg-neutral-900 border border-neutral-700 rounded-xl shadow-xl "
                "w-11/12 max-w-5xl max-h-[90vh] flex flex-col"
            ),
            children=[
                html.Div(
                    className="flex justify-between items-center py-3 px-4 border-b border-neutral-700 shrink-0",
                    children=[
                        html.H3("Architecture & Documentation",
                                className="font-semibold text-gray-200"),
                        html.Button(
                            "×", id="help-close-btn",
                            className=(
                                "w-8 h-8 text-2xl text-gray-400 hover:text-gray-200 "
                                "flex items-center justify-center rounded-full "
                                "hover:bg-neutral-800 transition-colors cursor-pointer leading-none"
                            ),
                        ),
                    ],
                ),
                html.Div(
                    className="p-5 overflow-y-auto",
                    children=html.Div(
                        className="prose prose-invert prose-sm max-w-none",
                        children=dcc.Markdown(_ARCH_MD, link_target="_blank"),
                    ),
                ),
            ],
        ),
    ),

    # ── Saved simulations modal ───────────────────────────────────────────────
    html.Div(
        id="saved-modal",
        className=_MODAL_CLOSED,
        children=html.Div(
            className=(
                "bg-neutral-900 border border-neutral-700 rounded-xl shadow-xl "
                "w-11/12 max-w-3xl max-h-[90vh] flex flex-col"
            ),
            children=[
                html.Div(
                    className="flex justify-between items-center py-3 px-4 border-b border-neutral-700 shrink-0",
                    children=[
                        html.H3("Saved simulations",
                                className="font-semibold text-gray-200"),
                        html.Button(
                            "×", id="saved-close-btn",
                            className=(
                                "w-8 h-8 text-2xl text-gray-400 hover:text-gray-200 "
                                "flex items-center justify-center rounded-full "
                                "hover:bg-neutral-800 transition-colors cursor-pointer leading-none"
                            ),
                        ),
                    ],
                ),
                html.Div(id="saved-list", className="p-4 overflow-y-auto"),
            ],
        ),
    ),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("poll-interval",  "disabled"),
    Output("active-dest",    "data"),
    Output("status-text",    "children"),
    Output("calc-btn",       "children"),
    Output("calc-btn",       "disabled"),
    Output("selected-cell",  "data"),
    Output("insertion-mode", "value"),
    Input("dest-dropdown",   "value"),
    State("n-azimuths",      "value"),
    State("n-elevations",    "value"),
    State("max-elevation",   "value"),
    State("n-speeds",        "value"),
    State("insertion-mode",  "value"),
    prevent_initial_call=True,
)
def on_destination_select(dest_id, n_az, n_el, max_el, n_sp, insertion_mode):
    if not dest_id:
        return True, None, "", "Calculate", True, None, "prograde"

    dest = ALL_DESTINATIONS[dest_id]
    default_mode = dest.default_insertion_mode
    cache_key = _make_cache_key(dest_id, n_az, n_el, max_el, n_sp, default_mode)
    with _lock:
        result = _compute_state.get("result")
        if result and result[6] == cache_key:
            return False, dest_id, "Loading cached result…", "Calculate", False, None, default_mode

    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")
    if os.path.exists(cache_path):
        threading.Thread(target=_run_computation,
                         args=(dest_id, n_az, n_el, max_el, n_sp, default_mode), daemon=True).start()
        return False, dest_id, "Loading from cache…", "Calculate", True, None, default_mode

    return True, dest_id, "", "Calculate", False, None, default_mode


@app.callback(
    Output("poll-interval", "disabled",     allow_duplicate=True),
    Output("status-text",   "children",     allow_duplicate=True),
    Output("calc-btn",      "children",     allow_duplicate=True),
    Output("calc-btn",      "disabled",     allow_duplicate=True),
    Output("selected-cell", "data",         allow_duplicate=True),
    Input("calc-btn",       "n_clicks"),
    State("active-dest",    "data"),
    State("n-azimuths",     "value"),
    State("n-elevations",   "value"),
    State("max-elevation",  "value"),
    State("n-speeds",       "value"),
    State("insertion-mode", "value"),
    prevent_initial_call=True,
)
def on_calculate_click(n_clicks, dest_id, n_az, n_el, max_el, n_sp, insertion_mode):
    if not dest_id:
        raise dash.exceptions.PreventUpdate

    threading.Thread(target=_run_computation,
                     args=(dest_id, n_az, n_el, max_el, n_sp, insertion_mode), daemon=True).start()
    return False, "Computing suitability map…", "Computing…", True, None


@app.callback(
    Output("moon-map",      "figure"),
    Output("traj-view",     "figure"),
    Output("progress-fill", "style"),
    Output("progress-wrap", "className"),
    Output("poll-interval", "disabled",  allow_duplicate=True),
    Output("status-text",   "children",  allow_duplicate=True),
    Output("calc-btn",      "children",  allow_duplicate=True),
    Output("calc-btn",      "disabled",  allow_duplicate=True),
    Input("poll-interval",  "n_intervals"),
    State("active-dest",    "data"),
    State("n-azimuths",     "value"),
    State("n-elevations",   "value"),
    State("max-elevation",  "value"),
    State("n-speeds",       "value"),
    State("insertion-mode", "value"),
    prevent_initial_call=True,
)
def poll_progress(n, dest_id, n_az, n_el, max_el, n_sp, insertion_mode):
    if not dest_id:
        return (build_empty_moon_map(), build_empty_trajectory_view(),
                {"width": "0%"}, _BAR_HIDDEN, True, "", "Calculate", True)

    cache_key = _make_cache_key(dest_id, n_az, n_el, max_el, n_sp, insertion_mode)

    with _lock:
        progress    = _compute_state["progress"]
        props_done  = _compute_state["props_done"]
        props_total = _compute_state["props_total"]
        sites_done  = _compute_state["sites_done"]
        sites_total = _compute_state["sites_total"]
        t_start     = _compute_state["t_start"]
        result      = _compute_state.get("result")
        error       = _compute_state.get("error")

    pct = int(progress * 100)

    if error:
        return (build_empty_moon_map("Computation failed"),
                build_empty_trajectory_view("Computation failed"),
                {"width": "0%"}, _BAR_HIDDEN, True,
                html.Span(f"Computation failed: {error}", className="text-red-400"),
                "Calculate", False)

    if result and result[6] == cache_key:
        dv_grid, trajs, az_grid, el_grid, spd_grid, _, _, cell_trajs = result
        dest = ALL_DESTINATIONS[dest_id]
        moon_fig = build_moon_map(LATS, LONS, dv_grid, dest.label,
                                  az_grid=az_grid, el_grid=el_grid, spd_grid=spd_grid)
        traj_fig = (build_trajectory_view(trajs, dest.label, uirevision=cache_key,
                                          target_orbits=dest.target_orbits(),
                                          selected_orbit_ids=_selected_orbit_ids(trajs))
                    if trajs else build_empty_trajectory_view("No valid trajectories found"))
        min_dv = np.nanmin(dv_grid[np.isfinite(dv_grid)]) if np.any(np.isfinite(dv_grid)) else 0
        return (moon_fig, traj_fig, {"width": "100%"}, _BAR_SHOWN, True,
                f"Done — best ΔV: {min_dv:.2f} km/s", "Calculate", False)

    elapsed = time.perf_counter() - t_start if t_start else 0
    if props_total:
        eta = ""
        if props_done > 0:
            remaining = elapsed / props_done * (props_total - props_done)
            eta = f"  ·  ~{_format_eta(remaining)} left"
        status = (f"Computing…  {props_done:,} / {props_total:,} propagations"
                  f"  ·  {sites_done} / {sites_total} sites"
                  f"  ·  {elapsed:.0f} s{eta}")
    else:
        status = "Computing…"

    return (build_empty_moon_map("Computing suitability map…"),
            build_empty_trajectory_view("Computing…"),
            {"width": f"{pct}%"}, _BAR_SHOWN, False, status, "Computing…", True)


@app.callback(
    Output("help-modal",    "className"),
    Input("help-btn",       "n_clicks"),
    Input("help-close-btn", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_help_modal(open_n, close_n):
    return _MODAL_OPEN if ctx.triggered_id == "help-btn" else _MODAL_CLOSED


@app.callback(
    Output("saved-modal", "className"),
    Output("saved-list",  "children"),
    Input("saved-btn",       "n_clicks"),
    Input("saved-close-btn", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_saved_modal(open_n, close_n):
    if ctx.triggered_id == "saved-btn":
        return _MODAL_OPEN, build_saved_list()
    return _MODAL_CLOSED, dash.no_update


@app.callback(
    Output("n-azimuths",    "value"),
    Output("n-elevations",  "value"),
    Output("max-elevation", "value"),
    Output("n-speeds",      "value"),
    Output("insertion-mode", "value",    allow_duplicate=True),
    Output("active-dest",    "data",     allow_duplicate=True),
    Output("selected-cell",  "data",     allow_duplicate=True),
    Output("poll-interval",  "disabled", allow_duplicate=True),
    Output("status-text",    "children", allow_duplicate=True),
    Output("calc-btn",       "children", allow_duplicate=True),
    Output("calc-btn",       "disabled", allow_duplicate=True),
    Output("saved-modal",    "className", allow_duplicate=True),
    Input({"type": "sim-open", "key": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def open_saved_sim(n_clicks_list):
    if not ctx.triggered_id or not any(n_clicks_list or []):
        raise dash.exceptions.PreventUpdate
    key = ctx.triggered_id["key"]
    settings = _parse_cache_key(key)
    if settings is None or settings["dest_id"] not in ALL_DESTINATIONS:
        raise dash.exceptions.PreventUpdate

    n_az, n_el, max_el, n_sp = (settings["n_az"], settings["n_el"],
                                settings["max_el"], settings["n_sp"])
    ins     = settings["insertion_mode"]
    dest_id = settings["dest_id"]

    threading.Thread(target=_run_computation,
                     args=(dest_id, n_az, n_el, max_el, n_sp, ins), daemon=True).start()
    return (n_az, n_el, max_el, n_sp, ins, dest_id, None, False,
            "Loading saved simulation…", "Calculate", True, _MODAL_CLOSED)


@app.callback(
    Output("saved-list", "children", allow_duplicate=True),
    Input({"type": "sim-delete", "key": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def delete_saved_sim(n_clicks_list):
    if not ctx.triggered_id or not any(n_clicks_list or []):
        raise dash.exceptions.PreventUpdate
    key = ctx.triggered_id["key"]
    for ext in (".npz", ".json"):
        try:
            os.remove(os.path.join(CACHE_DIR, f"{key}{ext}"))
        except FileNotFoundError:
            pass
    with _lock:
        result = _compute_state.get("result")
        if result and result[6] == key:
            _compute_state["result"] = None
    return build_saved_list()


@app.callback(
    Output("selected-cell", "data",         allow_duplicate=True),
    Input("moon-map",       "clickData"),
    State("selected-cell",  "data"),
    prevent_initial_call=True,
)
def on_map_click(click_data, current_sel):
    if not click_data:
        return None
    points = click_data.get("points", [])
    if not points:
        return None
    point = points[0]
    customdata = point.get("customdata")
    if customdata is None:
        return None
    lat, lon = customdata[0], customdata[1]
    if current_sel and current_sel["lat"] == lat and current_sel["lon"] == lon:
        return None
    return {"lat": lat, "lon": lon}


@app.callback(
    Output("traj-view",    "figure",        allow_duplicate=True),
    Output("moon-map",     "figure",        allow_duplicate=True),
    Input("selected-cell", "data"),
    State("active-dest",   "data"),
    State("n-azimuths",    "value"),
    State("n-elevations",  "value"),
    State("max-elevation", "value"),
    State("n-speeds",      "value"),
    State("insertion-mode", "value"),
    prevent_initial_call=True,
)
def render_selected(sel_cell, dest_id, n_az, n_el, max_el, n_sp, insertion_mode):
    if not dest_id:
        raise dash.exceptions.PreventUpdate
    cache_key = _make_cache_key(dest_id, n_az, n_el, max_el, n_sp, insertion_mode)
    with _lock:
        result = _compute_state.get("result")
    if not result or result[6] != cache_key:
        raise dash.exceptions.PreventUpdate

    dv_grid, trajs, az_grid, el_grid, spd_grid, _, _, cell_trajs = result
    dest = ALL_DESTINATIONS[dest_id]

    if sel_cell is None:
        moon_fig = build_moon_map(LATS, LONS, dv_grid, dest.label,
                                  az_grid=az_grid, el_grid=el_grid, spd_grid=spd_grid)
        if trajs:
            return build_trajectory_view(trajs, dest.label, uirevision=cache_key,
                                         target_orbits=dest.target_orbits(),
                                         selected_orbit_ids=_selected_orbit_ids(trajs)), moon_fig
        return build_empty_trajectory_view("No valid trajectories found"), moon_fig

    lat, lon = sel_cell["lat"], sel_cell["lon"]
    i = int(np.argmin(np.abs(LATS - lat)))
    j = int(np.argmin(np.abs(LONS - lon)))
    moon_fig = build_moon_map(LATS, LONS, dv_grid, dest.label,
                              az_grid=az_grid, el_grid=el_grid, spd_grid=spd_grid,
                              selected_ij=(i, j))
    if cell_trajs is not None and cell_trajs[i, j] is not None:
        label = f"{dest.label} — Lat {lat:.0f}°, Lon {lon:.0f}°"
        cell = [cell_trajs[i, j]]
        return build_trajectory_view(cell, label, uirevision=cache_key,
                                     target_orbits=dest.target_orbits(),
                                     selected_orbit_ids=_selected_orbit_ids(cell)), moon_fig
    return build_empty_trajectory_view("No trajectory data for this site"), moon_fig


@app.callback(
    Output("traj-view",         "figure",     allow_duplicate=True),
    Input("center-moon-btn",    "n_clicks"),
    Input("center-earth-btn",   "n_clicks"),
    State("active-dest",        "data"),
    State("selected-cell",      "data"),
    State("n-azimuths",         "value"),
    State("n-elevations",       "value"),
    State("max-elevation",      "value"),
    State("n-speeds",           "value"),
    State("insertion-mode",     "value"),
    prevent_initial_call=True,
)
def set_rotation_center(moon_n, earth_n, dest_id, sel_cell, n_az, n_el, max_el, n_sp, insertion_mode):
    from dash import ctx, Patch
    from physics.cr3bp import MU, DU_KM as _DU_KM

    if not dest_id:
        raise dash.exceptions.PreventUpdate

    cache_key = _make_cache_key(dest_id, n_az, n_el, max_el, n_sp, insertion_mode)
    with _lock:
        result = _compute_state.get("result")
    if not result or result[6] != cache_key:
        raise dash.exceptions.PreventUpdate

    cx, cy, cz, half = fixed_scene_bounds()

    if ctx.triggered_id == "center-moon-btn":
        tx = (1 - MU) * _DU_KM
    else:
        tx = -MU * _DU_KM

    patched = Patch()
    patched["layout"]["scene"]["camera"]["center"] = {
        "x": (tx - cx) / (2 * half),
        "y": (0.0 - cy) / (2 * half),
        "z": (0.0 - cz) / (2 * half),
    }
    return patched


@app.callback(
    Output("param-info",   "children"),
    Input("n-azimuths",    "value"),
    Input("n-elevations",  "value"),
    Input("max-elevation", "value"),
    Input("n-speeds",      "value"),
    Input("dest-dropdown", "value"),
)
def update_param_info(n_az, n_el, max_el, n_sp, dest_id):
    chip = _CHIP_CLS
    dest_name = ALL_DESTINATIONS[dest_id].label if dest_id else "none"
    _, elevations, _ = _make_sweep_arrays(n_az, n_el, max_el, n_sp)
    actual_n_el = len(elevations)
    n_props = int(n_az) * actual_n_el * int(n_sp)
    return [
        html.Span(f"dest: {dest_name}", className=chip),
        html.Span(f"grid: {len(LATS)}×{len(LONS)}", className=chip),
        html.Span(f"{int(n_az)} az × {actual_n_el} el × {int(n_sp)} spd", className=chip),
        html.Span(f"{n_props} prop/site", className=chip),
    ]
