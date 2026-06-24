"""
Lunar Mass Driver — Dash application module.

Imported by app.py (the entry point). Lives in its own module so that
ProcessPoolExecutor worker processes, which re-execute the __main__ script
(app.py) on Windows, never import this file and never pay the cost of
initialising Dash/Plotly.
"""

import os
import logging
import threading
import time
import numpy as np
import dash
from dash import dcc, html, Input, Output, State

from destinations.earth_leo import ALL_DESTINATIONS
from visualization.moon_map import build_moon_map, build_empty_moon_map
from visualization.trajectories import build_trajectory_view, build_empty_trajectory_view, scene_bounds, fixed_scene_bounds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lunar_sim")

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
# Far-side left (lon < -90°) is excluded for now; only the right side is tested.
LATS = np.array([-90, -60, -30,  0, 30, 60, 90], dtype=float)
LONS = np.array([-90, -60, -30,  0, 30, 60, 90, 120, 150, 180], dtype=float)

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


def _make_cache_key(dest_id, n_az, n_el, max_el, n_sp, insertion_mode="prograde"):
    return (f"{dest_id}_az{int(n_az)}_el{int(n_el)}x{int(max_el)}"
            f"_sp{int(n_sp)}_ins{insertion_mode}_g{len(LATS)}x{len(LONS)}")


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
    dv_grid, trajs, az_grid, el_grid, spd_grid, cell_trajs = compute_grid(
        LATS, LONS, dest,
        progress_cb=_progress,
        site_cb=_site_done,
        azimuths=azimuths,
        elevations=elevations,
        speeds_kms=speeds_kms,
    )
    logger.info("Grid computation finished in %.1f s", time.perf_counter() - t0)

    t_save = time.perf_counter()
    try:
        np.savez(cache_path, dv_grid=dv_grid, trajs=np.array(trajs, dtype=object),
                 az_grid=az_grid, el_grid=el_grid, spd_grid=spd_grid,
                 cell_trajs=cell_trajs)
        logger.info("Cache saved to %s (%.2f s)", cache_path, time.perf_counter() - t_save)
    except Exception as exc:
        logger.warning("Cache save failed: %s", exc)

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
                dcc.Input(id="n-azimuths", type="number", value=8, min=2, max=64, step=2,
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
                dcc.Input(id="n-speeds", type="number", value=10, min=3, max=60, step=1,
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
            return False, dest_id, "Loading cached result…", "Recalculate", False, None, default_mode

    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")
    if os.path.exists(cache_path):
        threading.Thread(target=_run_computation,
                         args=(dest_id, n_az, n_el, max_el, n_sp, default_mode), daemon=True).start()
        return False, dest_id, "Loading from cache…", "Recalculate", True, None, default_mode

    return True, dest_id, "", "Calculate", False, None, default_mode


@app.callback(
    Output("poll-interval", "disabled",     allow_duplicate=True),
    Output("status-text",   "children",     allow_duplicate=True),
    Output("calc-btn",      "children",     allow_duplicate=True),
    Output("calc-btn",      "disabled",     allow_duplicate=True),
    Output("selected-cell", "data",         allow_duplicate=True),
    Input("calc-btn",       "n_clicks"),
    State("active-dest",    "data"),
    State("calc-btn",       "children"),
    State("n-azimuths",     "value"),
    State("n-elevations",   "value"),
    State("max-elevation",  "value"),
    State("n-speeds",       "value"),
    State("insertion-mode", "value"),
    prevent_initial_call=True,
)
def on_calculate_click(n_clicks, dest_id, btn_label, n_az, n_el, max_el, n_sp, insertion_mode):
    if not dest_id:
        raise dash.exceptions.PreventUpdate

    if btn_label == "Recalculate":
        cache_key  = _make_cache_key(dest_id, n_az, n_el, max_el, n_sp, insertion_mode)
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")
        try:
            os.remove(cache_path)
        except FileNotFoundError:
            pass
        with _lock:
            _compute_state["result"] = None

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

    pct = int(progress * 100)

    if result and result[6] == cache_key:
        dv_grid, trajs, az_grid, el_grid, spd_grid, _, _, cell_trajs = result
        dest = ALL_DESTINATIONS[dest_id]
        moon_fig = build_moon_map(LATS, LONS, dv_grid, dest.label,
                                  az_grid=az_grid, el_grid=el_grid, spd_grid=spd_grid)
        traj_fig = (build_trajectory_view(trajs, dest.label, uirevision=cache_key)
                    if trajs else build_empty_trajectory_view("No valid trajectories found"))
        min_dv = np.nanmin(dv_grid[np.isfinite(dv_grid)]) if np.any(np.isfinite(dv_grid)) else 0
        return (moon_fig, traj_fig, {"width": "100%"}, _BAR_SHOWN, True,
                f"Done — best ΔV: {min_dv:.2f} km/s", "Recalculate", False)

    elapsed = time.perf_counter() - t_start if t_start else 0
    if props_total:
        status = (f"Computing…  {props_done:,} / {props_total:,} propagations"
                  f"  ·  {sites_done} / {sites_total} sites"
                  f"  ·  {elapsed:.0f} s")
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
    from dash import ctx
    return _MODAL_OPEN if ctx.triggered_id == "help-btn" else _MODAL_CLOSED


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
            return build_trajectory_view(trajs, dest.label, uirevision=cache_key), moon_fig
        return build_empty_trajectory_view("No valid trajectories found"), moon_fig

    lat, lon = sel_cell["lat"], sel_cell["lon"]
    i = int(np.argmin(np.abs(LATS - lat)))
    j = int(np.argmin(np.abs(LONS - lon)))
    moon_fig = build_moon_map(LATS, LONS, dv_grid, dest.label,
                              az_grid=az_grid, el_grid=el_grid, spd_grid=spd_grid,
                              selected_ij=(i, j))
    if cell_trajs is not None and cell_trajs[i, j] is not None:
        label = f"{dest.label} — Lat {lat:.0f}°, Lon {lon:.0f}°"
        return build_trajectory_view([cell_trajs[i, j]], label, uirevision=cache_key), moon_fig
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
    chip = ("px-2 py-0.5 rounded text-xs font-mono "
            "bg-neutral-800 text-gray-400 border border-neutral-700")
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
