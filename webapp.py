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
from visualization.trajectories import build_trajectory_view, build_empty_trajectory_view

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

LATS = np.arange(-60, 90, GRID_LAT_STEP, dtype=float)
LONS = np.arange(-150, 180, GRID_LON_STEP, dtype=float)

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
}
_lock = threading.Lock()


def _run_computation(dest_id):
    """Background thread: compute suitability grid and sample trajectories."""
    with _lock:
        _compute_state["running"]  = True
        _compute_state["progress"] = 0.0
        _compute_state["result"]   = None
        _compute_state["dest_id"]  = dest_id

    dest    = ALL_DESTINATIONS[dest_id]
    n_sites = len(LATS) * len(LONS)
    logger.info("Computation requested for '%s' (%d×%d = %d sites)",
                dest.label, len(LATS), len(LONS), n_sites)

    cache_path = os.path.join(CACHE_DIR, f"{dest_id}.npz")
    if os.path.exists(cache_path):
        logger.info("Cache hit: %s — loading from disk", cache_path)
        t0 = time.perf_counter()
        try:
            data    = np.load(cache_path, allow_pickle=True)
            dv_grid = data["dv_grid"]
            trajs   = list(data["trajs"])
            logger.info("Cache loaded in %.2f s", time.perf_counter() - t0)
            with _lock:
                _compute_state["result"]   = (dv_grid, trajs, dest_id)
                _compute_state["running"]  = False
                _compute_state["progress"] = 1.0
            return
        except Exception as exc:
            logger.warning("Cache load failed (%s) — recomputing", exc)

    from physics.optimizer import compute_grid, N_PROPS

    n_props_total = n_sites * N_PROPS
    with _lock:
        _compute_state["props_done"]  = 0
        _compute_state["props_total"] = n_props_total
        _compute_state["sites_done"]  = 0
        _compute_state["sites_total"] = n_sites
        _compute_state["t_start"]     = time.perf_counter()

    def _progress(props_done, props_total):
        with _lock:
            _compute_state["progress"]   = props_done / props_total
            _compute_state["props_done"] = props_done

    def _site_done(sites_done, sites_total):
        with _lock:
            _compute_state["sites_done"] = sites_done

    t0 = time.perf_counter()
    dv_grid, trajs = compute_grid(
        LATS, LONS, dest,
        progress_cb=_progress,
        site_cb=_site_done,
    )
    logger.info("Grid computation finished in %.1f s", time.perf_counter() - t0)

    t_save = time.perf_counter()
    try:
        np.savez(cache_path, dv_grid=dv_grid, trajs=np.array(trajs, dtype=object))
        logger.info("Cache saved to %s (%.2f s)", cache_path, time.perf_counter() - t_save)
    except Exception as exc:
        logger.warning("Cache save failed: %s", exc)

    with _lock:
        _compute_state["result"]   = (dv_grid, trajs, dest_id)
        _compute_state["running"]  = False
        _compute_state["progress"] = 1.0


# ── Dash app ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_scripts=[
        {"src": "https://cdn.tailwindcss.com"},
        {"src": "https://cdn.jsdelivr.net/npm/preline/dist/preline.js"},
    ],
    title="Lunar Mass Driver Sim",
    update_title=None,
)

DEST_OPTIONS = [{"label": d.label, "value": d.id} for d in ALL_DESTINATIONS.values()]

_BAR_HIDDEN = "hidden h-1.5 bg-neutral-800 rounded-full mb-1 overflow-hidden"
_BAR_SHOWN  = "h-1.5 bg-neutral-800 rounded-full mb-1 overflow-hidden"

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
                **{"data-hs-overlay": "#help-modal"},
            ),
        ]),
        html.Div(className="w-72", children=[
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
        html.Div(className="rounded-lg overflow-hidden",
                 children=dcc.Graph(id="traj-view", figure=build_empty_trajectory_view(),
                                    config={"displayModeBar": True,
                                            "modeBarButtonsToRemove": ["toImage"]})),
    ]),

    # ── Controls ──────────────────────────────────────────────────────────────
    html.Div(className="flex items-start gap-8 mt-3", children=[
        html.Div(className="w-80", children=[
            html.Label("Max launch elevation (°)", className="text-gray-500 text-xs"),
            dcc.Slider(id="max-elevation", min=0, max=10, step=1, value=5,
                       marks={i: str(i) for i in range(0, 11)},
                       tooltip={"placement": "bottom"},
                       className="mt-1"),
        ]),
        html.Div(id="param-info", className="text-gray-600 text-xs pt-6"),
    ]),

    # ── Hidden state ──────────────────────────────────────────────────────────
    dcc.Interval(id="poll-interval", interval=500, n_intervals=0, disabled=True),
    dcc.Store(id="active-dest", data=None),

    # ── Help modal (Preline overlay) ──────────────────────────────────────────
    html.Div(
        id="help-modal",
        className=(
            "hs-overlay hidden size-full fixed top-0 start-0 z-[80] "
            "overflow-x-hidden overflow-y-auto pointer-events-none"
        ),
        **{"role": "dialog", "aria-labelledby": "help-modal-label", "tabindex": "-1"},
        children=html.Div(
            className=(
                "hs-overlay-open:mt-7 hs-overlay-open:opacity-100 "
                "hs-overlay-open:duration-500 mt-0 opacity-0 ease-out transition-all "
                "lg:max-w-5xl lg:w-full m-4 lg:mx-auto"
            ),
            children=html.Div(
                className=(
                    "flex flex-col bg-neutral-900 border border-neutral-700 "
                    "rounded-xl shadow-xl pointer-events-auto"
                ),
                children=[
                    html.Div(
                        className="flex justify-between items-center py-3 px-4 border-b border-neutral-700",
                        children=[
                            html.H3("Architecture & Documentation",
                                    id="help-modal-label",
                                    className="font-semibold text-gray-200"),
                            html.Button(
                                "×",
                                className=(
                                    "w-8 h-8 text-2xl text-gray-400 hover:text-gray-200 "
                                    "flex items-center justify-center rounded-full "
                                    "hover:bg-neutral-800 transition-colors cursor-pointer leading-none"
                                ),
                                **{"data-hs-overlay": "#help-modal"},
                            ),
                        ],
                    ),
                    html.Div(
                        className="p-5 overflow-y-auto max-h-[75vh]",
                        children=html.Div(
                            dcc.Markdown(_ARCH_MD, link_target="_blank"),
                            className="prose prose-invert prose-sm max-w-none",
                        ),
                    ),
                ],
            ),
        ),
    ),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("poll-interval", "disabled"),
    Output("active-dest",   "data"),
    Output("status-text",   "children"),
    Input("dest-dropdown",  "value"),
    Input("max-elevation",  "value"),
    prevent_initial_call=True,
)
def on_destination_select(dest_id, max_el):
    if not dest_id:
        return True, None, ""

    with _lock:
        result = _compute_state.get("result")
        if result and result[2] == dest_id:
            return True, dest_id, "Cached result loaded."

    threading.Thread(target=_run_computation, args=(dest_id,), daemon=True).start()
    return False, dest_id, "Computing suitability map…"


@app.callback(
    Output("moon-map",      "figure"),
    Output("traj-view",     "figure"),
    Output("progress-fill", "style"),
    Output("progress-wrap", "className"),
    Output("poll-interval", "disabled",  allow_duplicate=True),
    Output("status-text",   "children",  allow_duplicate=True),
    Input("poll-interval",  "n_intervals"),
    State("active-dest",    "data"),
    prevent_initial_call=True,
)
def poll_progress(n, dest_id):
    if not dest_id:
        return (build_empty_moon_map(), build_empty_trajectory_view(),
                {"width": "0%"}, _BAR_HIDDEN, True, "")

    with _lock:
        progress    = _compute_state["progress"]
        props_done  = _compute_state["props_done"]
        props_total = _compute_state["props_total"]
        sites_done  = _compute_state["sites_done"]
        sites_total = _compute_state["sites_total"]
        t_start     = _compute_state["t_start"]
        result      = _compute_state.get("result")

    pct = int(progress * 100)

    if result and result[2] == dest_id:
        dv_grid, trajs, _ = result
        dest = ALL_DESTINATIONS[dest_id]
        moon_fig = build_moon_map(LATS, LONS, dv_grid, dest.label)
        traj_fig = (build_trajectory_view(trajs, dest.label)
                    if trajs else build_empty_trajectory_view("No valid trajectories found"))
        min_dv = np.nanmin(dv_grid[np.isfinite(dv_grid)]) if np.any(np.isfinite(dv_grid)) else 0
        return moon_fig, traj_fig, {"width": "100%"}, _BAR_SHOWN, True, f"Done — best ΔV: {min_dv:.2f} km/s"

    elapsed = time.perf_counter() - t_start if t_start else 0
    if props_total:
        status = (f"Computing…  {props_done:,} / {props_total:,} propagations"
                  f"  ·  {sites_done} / {sites_total} sites"
                  f"  ·  {elapsed:.0f} s")
    else:
        status = "Computing…"

    return (build_empty_moon_map("Computing suitability map…"),
            build_empty_trajectory_view("Computing…"),
            {"width": f"{pct}%"}, _BAR_SHOWN, False, status)


@app.callback(
    Output("param-info",   "children"),
    Input("max-elevation", "value"),
    Input("dest-dropdown", "value"),
)
def update_param_info(max_el, dest_id):
    from physics.optimizer import N_AZ, ELEVATIONS, SPEEDS_KMS, N_PROPS
    dest_name = ALL_DESTINATIONS[dest_id].label if dest_id else "—"
    return (f"Destination: {dest_name} | "
            f"Grid: {len(LATS)}×{len(LONS)} sites | "
            f"Sweep: {N_AZ} azimuths × {len(ELEVATIONS)} elevations × {len(SPEEDS_KMS)} speeds"
            f" = {N_PROPS} propagations/site")
