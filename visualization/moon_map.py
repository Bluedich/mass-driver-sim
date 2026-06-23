"""
Moon surface suitability map — equirectangular projection.

Shows the Moon satellite photo as background and overlays a semi-transparent
green-to-red heatmap of post-launch ΔV.
"""

import numpy as np
import plotly.graph_objects as go
import base64
import os

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


def _load_image_b64(filename):
    path = os.path.join(ASSET_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _add_arrow_overlay(fig, lats, lons, dv_grid, az_grid, el_grid, spd_grid,
                       selected_ij=None):
    """Add launch-direction arrows for every feasible grid site."""
    GRID_STEP = 30.0   # degrees — matches GRID_LAT_STEP / GRID_LON_STEP in webapp.py
    MAX_HALF  = GRID_STEP * 0.40   # 12° — arrow length at 0° elevation
    MIN_HALF  = GRID_STEP * 0.10   #  3° — arrow length at 90° elevation

    norm_shaft_x, norm_shaft_y = [], []
    norm_tip_x,   norm_tip_y,   norm_tip_az = [], [], []
    sel_shaft_x,  sel_shaft_y  = [], []
    sel_tip_x,    sel_tip_y,    sel_tip_az  = [], [], []
    mid_x,        mid_y        = [], []
    customdata                 = []

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            dv  = dv_grid[i, j]
            az  = az_grid[i, j]
            el  = el_grid[i, j]
            spd = spd_grid[i, j] if spd_grid is not None else np.nan
            if not (np.isfinite(dv) and np.isfinite(az) and np.isfinite(el)):
                continue

            half   = MIN_HALF + (MAX_HALF - MIN_HALF) * (1.0 - el / 90.0)
            az_rad = np.deg2rad(az)
            dx     = half * np.sin(az_rad)   # east (+lon)
            dy     = half * np.cos(az_rad)   # north (+lat)

            is_sel = selected_ij is not None and selected_ij == (i, j)
            if is_sel:
                sel_shaft_x += [lon - dx, lon + dx, None]
                sel_shaft_y += [lat - dy, lat + dy, None]
                sel_tip_x.append(lon + dx)
                sel_tip_y.append(lat + dy)
                sel_tip_az.append(az)
            else:
                norm_shaft_x += [lon - dx, lon + dx, None]
                norm_shaft_y += [lat - dy, lat + dy, None]
                norm_tip_x.append(lon + dx)
                norm_tip_y.append(lat + dy)
                norm_tip_az.append(az)

            mid_x.append(lon)
            mid_y.append(lat)
            customdata.append([lat, lon, az, el, spd, dv])

    if not mid_x:
        return

    # Normal arrows
    if norm_shaft_x:
        fig.add_trace(go.Scatter(
            x=norm_shaft_x, y=norm_shaft_y,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.85)", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=norm_tip_x, y=norm_tip_y,
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=8,
                color="rgba(255,255,255,0.9)",
                angle=norm_tip_az,
                line=dict(width=0),
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Selected arrow (gold, thicker)
    if sel_shaft_x:
        fig.add_trace(go.Scatter(
            x=sel_shaft_x, y=sel_shaft_y,
            mode="lines",
            line=dict(color="rgba(255,210,0,1.0)", width=3),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=sel_tip_x, y=sel_tip_y,
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color="rgba(255,210,0,1.0)",
                angle=sel_tip_az,
                line=dict(color="white", width=1),
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Invisible hit-targets for hover/click (all cells)
    fig.add_trace(go.Scatter(
        x=mid_x, y=mid_y,
        mode="markers",
        marker=dict(size=18, opacity=0, color="white"),
        customdata=customdata,
        hovertemplate=(
            "<b>Lat %{customdata[0]:.0f}°, Lon %{customdata[1]:.0f}°</b><br>"
            "Azimuth: %{customdata[2]:.1f}° (CW from N)<br>"
            "Elevation: %{customdata[3]:.1f}°<br>"
            "Launch speed: %{customdata[4]:.2f} km/s<br>"
            "Post-launch ΔV: %{customdata[5]:.3f} km/s"
            "<extra></extra>"
        ),
        showlegend=False,
    ))


def build_moon_map(lats, lons, dv_grid, destination_label="",
                   az_grid=None, el_grid=None, spd_grid=None, selected_ij=None):
    """
    Build a Plotly figure: equirectangular Moon map with ΔV heatmap overlay.

    Parameters
    ----------
    lats : 1-D array of latitudes (degrees, -90 to 90)
    lons : 1-D array of longitudes (degrees, -180 to 180)
    dv_grid : 2-D array shape (len(lats), len(lons)), ΔV in km/s
    destination_label : str

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # ── Background: Moon satellite photo ──────────────────────────────────────
    moon_b64 = _load_image_b64("moon_surface.jpg")
    if moon_b64:
        fig.add_layout_image(
            dict(
                source=f"data:image/jpeg;base64,{moon_b64}",
                xref="x", yref="y",
                x=-180, y=90,
                sizex=360, sizey=180,
                sizing="stretch",
                opacity=1.0,
                layer="below",
            )
        )

    # ── ΔV heatmap overlay ────────────────────────────────────────────────────
    # Replace inf with NaN so Plotly renders them transparent
    dv_plot = np.where(np.isfinite(dv_grid), dv_grid, np.nan)

    # Custom green→yellow→red colorscale
    colorscale = [
        [0.0, "rgb(0,180,0)"],
        [0.3, "rgb(100,220,0)"],
        [0.5, "rgb(255,220,0)"],
        [0.7, "rgb(255,140,0)"],
        [1.0, "rgb(200,0,0)"],
    ]

    fig.add_trace(go.Heatmap(
        x=lons,
        y=lats,
        z=dv_plot,
        colorscale=colorscale,
        opacity=0.65,
        zmin=np.nanpercentile(dv_plot, 2)  if np.any(np.isfinite(dv_plot)) else 0,
        zmax=np.nanpercentile(dv_plot, 98) if np.any(np.isfinite(dv_plot)) else 5,
        colorbar=dict(
            title=dict(text="Post-launch ΔV (km/s)", side="right"),
            thickness=14,
            len=0.8,
        ),
        hoverongaps=False,
        hovertemplate="Lon: %{x:.1f}°<br>Lat: %{y:.1f}°<br>ΔV: %{z:.2f} km/s<extra></extra>",
    ))

    if az_grid is not None and el_grid is not None:
        _add_arrow_overlay(fig, lats, lons, dv_grid, az_grid, el_grid, spd_grid,
                           selected_ij=selected_ij)

    fig.update_layout(
        title=dict(text=f"Launch suitability — {destination_label}", x=0.5,
                   font=dict(size=13, color="#ccc")),
        xaxis=dict(
            title="Longitude (°)", range=[-180, 180],
            tickvals=[-180, -120, -60, 0, 60, 120, 180],
            gridcolor="#333", zerolinecolor="#555",
            color="#aaa",
        ),
        yaxis=dict(
            title="Latitude (°)", range=[-90, 90],
            tickvals=[-90, -60, -30, 0, 30, 60, 90],
            gridcolor="#333", zerolinecolor="#555",
            color="#aaa", scaleanchor="x", scaleratio=1,
        ),
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        margin=dict(l=50, r=20, t=40, b=40),
        height=420,
    )

    return fig


def build_empty_moon_map(message="Select a destination to compute suitability map"):
    """Placeholder figure shown before computation."""
    moon_b64 = _load_image_b64("moon_surface.jpg")
    fig = go.Figure()

    if moon_b64:
        fig.add_layout_image(
            dict(
                source=f"data:image/jpeg;base64,{moon_b64}",
                xref="x", yref="y",
                x=-180, y=90,
                sizex=360, sizey=180,
                sizing="stretch",
                opacity=0.7,
                layer="below",
            )
        )

    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#aaa"),
    )

    fig.update_layout(
        xaxis=dict(range=[-180, 180], showgrid=False, zeroline=False,
                   showticklabels=True, color="#aaa",
                   tickvals=[-180, -120, -60, 0, 60, 120, 180]),
        yaxis=dict(range=[-90, 90], showgrid=False, zeroline=False,
                   showticklabels=True, color="#aaa",
                   tickvals=[-90, -60, -30, 0, 30, 60, 90],
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        margin=dict(l=50, r=20, t=40, b=40),
        height=420,
    )
    return fig
