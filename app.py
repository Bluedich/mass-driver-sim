"""
Lunar Mass Driver Orbital Simulation — Dash webapp.

Run:
    python app.py
Then open http://localhost:8050
"""

import os
import logging
import threading
import time
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

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

# ── Grid resolution ────────────────────────────────────────────────────────────
# 30° grid → ~55 sites, ~8 min serial.  Cached after first run.
GRID_LAT_STEP = 30
GRID_LON_STEP = 30

LATS = np.arange(-60, 90, GRID_LAT_STEP, dtype=float)
LONS = np.arange(-150, 180, GRID_LON_STEP, dtype=float)

# ── Shared computation state ───────────────────────────────────────────────────
_compute_state = {
    "running":   False,
    "progress":  0.0,    # 0.0 – 1.0
    "result":    None,   # (dv_grid, trajs, dest_id) when done
    "dest_id":   None,
}
_lock = threading.Lock()


def _run_computation(dest_id):
    """Background thread: compute suitability grid and sample trajectories."""
    with _lock:
        _compute_state["running"]  = True
        _compute_state["progress"] = 0.0
        _compute_state["result"]   = None
        _compute_state["dest_id"]  = dest_id

    dest  = ALL_DESTINATIONS[dest_id]
    n_sites = len(LATS) * len(LONS)
    logger.info("Computation requested for '%s' (%d×%d = %d sites)",
                dest.label, len(LATS), len(LONS), n_sites)

    # Check cache
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

    from physics.optimizer import compute_grid

    def _progress(frac):
        with _lock:
            _compute_state["progress"] = frac

    t0 = time.perf_counter()
    dv_grid, trajs = compute_grid(LATS, LONS, dest, progress_cb=_progress)
    elapsed = time.perf_counter() - t0
    logger.info("Grid computation finished in %.1f s", elapsed)

    # Save cache
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
    external_stylesheets=[dbc.themes.DARKLY],
    title="Lunar Mass Driver Sim",
)

DEST_OPTIONS = [
    {"label": d.label, "value": d.id}
    for d in ALL_DESTINATIONS.values()
]

app.layout = dbc.Container(fluid=True, style={"backgroundColor": "#0d0d0d", "minHeight": "100vh",
                                               "padding": "12px"}, children=[
    # ── Header ────────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(html.Div([
            html.H4("Lunar Mass Driver — Orbital Suitability",
                    style={"color": "#ddd", "marginBottom": "0"}),
            dbc.Button("?", id="help-btn", size="sm", color="secondary", outline=True,
                       title="Architecture & documentation",
                       style={"borderRadius": "50%", "width": "26px", "height": "26px",
                              "padding": "0", "fontSize": "13px", "flexShrink": "0"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}), width=8),
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Label("Destination", style={"color": "#aaa", "fontSize": "12px",
                                                          "marginBottom": "2px"}), width=12),
                dbc.Col(dcc.Dropdown(
                    id="dest-dropdown",
                    options=DEST_OPTIONS,
                    value=None,
                    placeholder="Select destination…",
                    clearable=False,
                    style={"backgroundColor": "#1e1e1e", "color": "#eee",
                           "border": "1px solid #444"},
                ), width=12),
            ])
        ], width=4),
    ], align="center", style={"marginBottom": "10px"}),

    # ── Progress bar (hidden when idle) ───────────────────────────────────────
    dbc.Row([
        dbc.Col([
            dbc.Progress(id="progress-bar", value=0, max=100,
                         color="success", striped=True, animated=True,
                         style={"height": "6px", "display": "none"},
                         className="mb-1"),
            html.Div(id="status-text",
                     style={"color": "#888", "fontSize": "11px", "height": "14px"}),
        ], width=12),
    ]),

    # ── Main panels ───────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dcc.Graph(id="moon-map",
                          figure=build_empty_moon_map(),
                          config={"displayModeBar": False},
                          style={"borderRadius": "6px", "overflow": "hidden"}),
                width=6),
        dbc.Col(dcc.Graph(id="traj-view",
                          figure=build_empty_trajectory_view(),
                          config={"displayModeBar": True,
                                  "modeBarButtonsToRemove": ["toImage"]},
                          style={"borderRadius": "6px", "overflow": "hidden"}),
                width=6),
    ], style={"marginTop": "6px"}),

    # ── Controls row ──────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Label("Max launch elevation (°)", style={"color": "#888", "fontSize": "11px"}),
            dcc.Slider(id="max-elevation", min=0, max=10, step=1, value=5,
                       marks={i: str(i) for i in range(0, 11)},
                       tooltip={"placement": "bottom"},
                       className="mt-1"),
        ], width=4),
        dbc.Col(html.Div(id="param-info",
                         style={"color": "#666", "fontSize": "11px", "paddingTop": "24px"}),
                width=8),
    ], style={"marginTop": "10px"}),

    # ── Polling interval ──────────────────────────────────────────────────────
    dcc.Interval(id="poll-interval", interval=500, n_intervals=0, disabled=True),

    # ── Hidden stores ─────────────────────────────────────────────────────────
    dcc.Store(id="active-dest", data=None),

    # ── Help modal ────────────────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Architecture & Documentation"),
                        close_button=True),
        dbc.ModalBody(dcc.Markdown(_ARCH_MD, link_target="_blank")),
    ], id="help-modal", size="xl", scrollable=True, is_open=False),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("poll-interval", "disabled"),
    Output("active-dest", "data"),
    Output("status-text", "children"),
    Input("dest-dropdown", "value"),
    Input("max-elevation", "value"),
    prevent_initial_call=True,
)
def on_destination_select(dest_id, max_el):
    if not dest_id:
        return True, None, ""

    # Check if already cached in memory
    with _lock:
        result = _compute_state.get("result")
        if result and result[2] == dest_id:
            return True, dest_id, "Cached result loaded."

    # Start background computation
    t = threading.Thread(target=_run_computation, args=(dest_id,), daemon=True)
    t.start()

    return False, dest_id, "Computing suitability map…"


@app.callback(
    Output("moon-map",     "figure"),
    Output("traj-view",    "figure"),
    Output("progress-bar", "value"),
    Output("progress-bar", "style"),
    Output("poll-interval","disabled", allow_duplicate=True),
    Output("status-text",  "children", allow_duplicate=True),
    Input("poll-interval", "n_intervals"),
    State("active-dest",   "data"),
    prevent_initial_call=True,
)
def poll_progress(n, dest_id):
    if not dest_id:
        return (build_empty_moon_map(), build_empty_trajectory_view(),
                0, {"display": "none"}, True, "")

    with _lock:
        running  = _compute_state["running"]
        progress = _compute_state["progress"]
        result   = _compute_state.get("result")

    bar_val   = int(progress * 100)
    bar_style = {"height": "6px", "display": "block"}

    if result and result[2] == dest_id:
        dv_grid, trajs, _ = result
        dest = ALL_DESTINATIONS[dest_id]

        moon_fig = build_moon_map(LATS, LONS, dv_grid, dest.label)
        traj_fig = (build_trajectory_view(trajs, dest.label)
                    if trajs else build_empty_trajectory_view("No valid trajectories found"))

        min_dv = np.nanmin(dv_grid[np.isfinite(dv_grid)]) if np.any(np.isfinite(dv_grid)) else 0
        status = f"Done — best ΔV: {min_dv:.2f} km/s"
        return moon_fig, traj_fig, 100, {"height": "6px", "display": "block"}, True, status

    status = f"Computing… {bar_val}%"
    return (build_empty_moon_map("Computing suitability map…"),
            build_empty_trajectory_view("Computing…"),
            bar_val, bar_style, False, status)


@app.callback(
    Output("param-info", "children"),
    Input("max-elevation", "value"),
    Input("dest-dropdown", "value"),
)
def update_param_info(max_el, dest_id):
    dest_name = ALL_DESTINATIONS[dest_id].label if dest_id else "—"
    return (f"Destination: {dest_name} | "
            f"Max elevation: {max_el}° | "
            f"Grid: {len(LATS)}×{len(LONS)} sites | "
            f"Azimuth sweep: 36 steps × 6 elevations")


@app.callback(
    Output("help-modal", "is_open"),
    Input("help-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_help_modal(_):
    return True


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
