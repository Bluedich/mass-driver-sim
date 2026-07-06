"""
Moon surface suitability map — equirectangular projection.

Shows the Moon satellite photo as background and overlays a semi-transparent
green-to-red heatmap of post-launch ΔV.  Launch sites are equal-area tiles
(see physics.tiling); the map rasterizes each pixel to its nearest tile, so
tiles render as flat-colored regions with no borders unless the border overlay
is toggled on.
"""

import numpy as np
import plotly.graph_objects as go
import base64
import os

from physics.tiling import latlon_to_xyz, split_antimeridian

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

# Raster resolution for the nearest-tile heatmap (equirectangular pixels).
_RASTER_NLON = 720
_RASTER_NLAT = 360


def _load_image_b64(filename):
    path = os.path.join(ASSET_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _rasterize(tiling, dv):
    """Nearest-tile ΔV raster over an equirectangular pixel grid.

    Returns (x_disp, y_lat, z) for a go.Heatmap: each pixel takes the ΔV of the
    tile whose centre is nearest (great-circle) — flat-colored regions, no edges.
    Unreachable tiles map to NaN (transparent).
    """
    lon = np.linspace(-180.0, 180.0, _RASTER_NLON)
    lat = np.linspace(-90.0, 90.0, _RASTER_NLAT)
    LO, LA = np.meshgrid(lon, lat)
    xyz = latlon_to_xyz(LA.ravel(), LO.ravel())
    idx = tiling.nearest_index(xyz)
    z = np.asarray(dv, dtype=float)[idx].reshape(_RASTER_NLAT, _RASTER_NLON)
    z = np.where(np.isfinite(z), z, np.nan)
    # x = −lon so east renders on the left, matching the photo convention.
    return -lon, lat, z


def _add_borders(fig, tiling):
    """Overlay the Voronoi cell edges as thin grey lines (dateline-split)."""
    xs, ys = [], []
    for ring in tiling.polygons_latlon:
        blat, blon = split_antimeridian(ring[:, 0], ring[:, 1])
        xs.extend((-blon).tolist())
        ys.extend(blat.tolist())
        xs.append(None)
        ys.append(None)
    if not xs:
        return
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color="rgba(255,255,255,0.35)", width=0.8),
        hoverinfo="skip", showlegend=False,
    ))


def _add_arrow_overlay(fig, tiling, dv, az, el, spd, selected_tile=None):
    """Add launch-direction arrows + click/hover hit-targets, one per feasible tile."""
    n = tiling.n
    # Arrow length scales with the characteristic tile spacing (≈√(41253/N)°),
    # so arrows stay proportional to cell size as density changes.
    spacing = float(np.sqrt(41253.0 / max(1, n)))
    MAX_HALF = spacing * 0.45   # arrow half-length at 0° elevation
    MIN_HALF = spacing * 0.12   # arrow half-length at 90° elevation

    norm_shaft_x, norm_shaft_y = [], []
    norm_tip_x,   norm_tip_y,   norm_tip_az = [], [], []
    sel_shaft_x,  sel_shaft_y  = [], []
    sel_tip_x,    sel_tip_y,    sel_tip_az  = [], [], []
    mid_x,        mid_y        = [], []
    customdata                 = []

    for k in range(n):
        lat, lon = tiling.centers_latlon[k]
        dvk  = dv[k]
        azk  = az[k]
        elk  = el[k]
        spdk = spd[k] if spd is not None else np.nan
        if not (np.isfinite(dvk) and np.isfinite(azk) and np.isfinite(elk)):
            continue

        half   = MIN_HALF + (MAX_HALF - MIN_HALF) * (1.0 - elk / 90.0)
        az_rad = np.deg2rad(azk)
        dx     = half * np.sin(az_rad)   # east (+lon)
        dy     = half * np.cos(az_rad)   # north (+lat)

        xc = -lon   # display x (east on the left)
        az_disp = (360 - azk) % 360      # mirror azimuth for the flipped x-axis

        is_sel = selected_tile is not None and selected_tile == k
        if is_sel:
            sel_shaft_x += [xc + dx, xc - dx, None]
            sel_shaft_y += [lat - dy, lat + dy, None]
            sel_tip_x.append(xc - dx)
            sel_tip_y.append(lat + dy)
            sel_tip_az.append(az_disp)
        else:
            norm_shaft_x += [xc + dx, xc - dx, None]
            norm_shaft_y += [lat - dy, lat + dy, None]
            norm_tip_x.append(xc - dx)
            norm_tip_y.append(lat + dy)
            norm_tip_az.append(az_disp)

        mid_x.append(xc)
        mid_y.append(lat)
        customdata.append([lat, lon, azk, elk, spdk, dvk, k])

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

    # Invisible hit-targets for hover/click (one per feasible tile).
    # customdata[6] carries the tile index, used by the click callback.
    fig.add_trace(go.Scatter(
        x=mid_x, y=mid_y,
        mode="markers",
        marker=dict(size=16, opacity=0, color="white"),
        customdata=customdata,
        hovertemplate=(
            "<b>Lat %{customdata[0]:.1f}°, Lon %{customdata[1]:.1f}°</b><br>"
            "Azimuth: %{customdata[2]:.1f}° (CW from N)<br>"
            "Elevation: %{customdata[3]:.1f}°<br>"
            "Launch speed: %{customdata[4]:.2f} km/s<br>"
            "Post-launch ΔV: %{customdata[5]:.3f} km/s"
            "<extra></extra>"
        ),
        showlegend=False,
    ))


def build_moon_map(tiling, dv, destination_label="",
                   az=None, el=None, spd=None, selected_tile=None,
                   show_borders=False):
    """
    Build a Plotly figure: equirectangular Moon map with an equal-area ΔV heatmap.

    Parameters
    ----------
    tiling : physics.tiling.Tiling — the equal-area launch-site tiling
    dv  : 1-D array (N,), post-launch ΔV in km/s per tile (inf → transparent)
    az, el, spd : 1-D arrays (N,), best launch params per tile (for arrows/hover)
    selected_tile : int or None — tile to highlight in gold
    show_borders : bool — overlay the Voronoi cell edges when True

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

    # ── ΔV heatmap overlay (nearest-tile raster) ──────────────────────────────
    x_disp, y_lat, z = _rasterize(tiling, dv)

    # Custom green→yellow→red colorscale
    colorscale = [
        [0.0, "rgb(0,180,0)"],
        [0.3, "rgb(100,220,0)"],
        [0.5, "rgb(255,220,0)"],
        [0.7, "rgb(255,140,0)"],
        [1.0, "rgb(200,0,0)"],
    ]

    finite = np.any(np.isfinite(z))
    fig.add_trace(go.Heatmap(
        x=x_disp,
        y=y_lat,
        z=z,
        colorscale=colorscale,
        opacity=0.65,
        zmin=np.nanpercentile(z, 2)  if finite else 0,
        zmax=np.nanpercentile(z, 98) if finite else 5,
        colorbar=dict(
            title=dict(text="Post-launch ΔV (km/s)", side="right"),
            thickness=14,
            len=0.8,
        ),
        hoverinfo="skip",
    ))

    if show_borders:
        _add_borders(fig, tiling)

    if az is not None and el is not None:
        _add_arrow_overlay(fig, tiling, dv, az, el, spd, selected_tile=selected_tile)

    fig.update_layout(
        title=dict(text=f"Launch suitability — {destination_label}", x=0.5,
                   font=dict(size=13, color="#ccc")),
        xaxis=dict(
            title="Longitude (°)",
            range=[-180, 180],
            # x = −lon: east is on the left, west on the right
            tickvals=[-180, -120, -60, 0, 60, 120, 180],
            ticktext=["180°", "120°E", "60°E", "0°", "60°W", "120°W", "180°"],
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
                   tickvals=[-180, -120, -60, 0, 60, 120, 180],
                   ticktext=["180°", "120°E", "60°E", "0°", "60°W", "120°W", "180°"]),
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
