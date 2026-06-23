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


def build_moon_map(lats, lons, dv_grid, destination_label=""):
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
