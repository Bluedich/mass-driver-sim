"""
3-D trajectory visualisation in the CR3BP rotating frame.

Shows:
  - Earth sphere (blue)
  - Moon sphere (grey)
  - Sample transfer trajectories (coloured lines)
  - Burn locations (dots, green→red by ΔV magnitude)
"""

import numpy as np
import plotly.graph_objects as go
import base64, os

from physics.cr3bp import MU, R_EARTH_DU, R_MOON_DU, DU_KM

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


def _sphere_mesh(cx, cy, cz, radius, n=24):
    """Return x,y,z meshes for a sphere centred at (cx,cy,cz)."""
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = cx + radius * np.outer(np.cos(u), np.sin(v))
    y = cy + radius * np.outer(np.sin(u), np.sin(v))
    z = cz + radius * np.outer(np.ones(n), np.cos(v))
    return x, y, z


def _load_image_b64(filename):
    path = os.path.join(ASSET_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _dv_color(dv_kms, dv_min, dv_max):
    """Map a ΔV value to an RGB string (green=low, red=high)."""
    if dv_max <= dv_min:
        t = 0.5
    else:
        t = np.clip((dv_kms - dv_min) / (dv_max - dv_min), 0, 1)
    r = int(255 * t)
    g = int(255 * (1 - t))
    return f"rgb({r},{g},0)"


def build_trajectory_view(trajectories, destination_label=""):
    """
    Build a Plotly 3-D figure showing trajectories in the CR3BP rotating frame.

    Parameters
    ----------
    trajectories : list of dicts, each with keys:
        't', 'x', 'y', 'z' : arrays (non-dim DU)
        'burns' : list of {'x','y','z','dv_kms'}
    destination_label : str

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # ── Earth sphere ──────────────────────────────────────────────────────────
    ex, ey, ez = _sphere_mesh(-MU, 0, 0, R_EARTH_DU * 3, n=30)
    earth_b64 = _load_image_b64("earth.jpg")

    fig.add_trace(go.Surface(
        x=ex * DU_KM, y=ey * DU_KM, z=ez * DU_KM,
        colorscale=[[0, "rgb(30,80,200)"], [1, "rgb(30,80,200)"]],
        showscale=False,
        opacity=1.0,
        name="Earth",
        hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.8),
    ))

    # ── Moon sphere ───────────────────────────────────────────────────────────
    mx, my, mz = _sphere_mesh(1 - MU, 0, 0, R_MOON_DU * 3, n=24)
    fig.add_trace(go.Surface(
        x=mx * DU_KM, y=my * DU_KM, z=mz * DU_KM,
        colorscale=[[0, "rgb(160,160,160)"], [1, "rgb(220,220,220)"]],
        showscale=False,
        opacity=1.0,
        name="Moon",
        hoverinfo="skip",
        lighting=dict(ambient=0.7, diffuse=0.6),
    ))

    # ── Trajectories + burns ──────────────────────────────────────────────────
    all_dv = []
    for traj in trajectories:
        for burn in traj.get("burns", []):
            if np.isfinite(burn["dv_kms"]):
                all_dv.append(burn["dv_kms"])

    dv_min = min(all_dv) if all_dv else 0.0
    dv_max = max(all_dv) if all_dv else 1.0

    colors = [
        "#4fc3f7", "#81d4fa", "#b3e5fc",
        "#80cbc4", "#a5d6a7", "#fff176",
        "#ffcc80", "#ef9a9a", "#ce93d8",
        "#b0bec5", "#90a4ae", "#78909c",
    ]

    for i, traj in enumerate(trajectories):
        color = colors[i % len(colors)]
        x_km = traj["x"] * DU_KM
        y_km = traj["y"] * DU_KM
        z_km = traj["z"] * DU_KM

        fig.add_trace(go.Scatter3d(
            x=x_km, y=y_km, z=z_km,
            mode="lines",
            line=dict(color=color, width=2),
            name=f"Trajectory {i+1}",
            hoverinfo="skip",
            showlegend=False,
        ))

        # Burn dots
        for burn in traj.get("burns", []):
            bcolor = _dv_color(burn["dv_kms"], dv_min, dv_max)
            fig.add_trace(go.Scatter3d(
                x=[burn["x"] * DU_KM],
                y=[burn["y"] * DU_KM],
                z=[burn["z"] * DU_KM],
                mode="markers",
                marker=dict(size=8, color=bcolor,
                            line=dict(color="white", width=1)),
                hovertemplate=f"Burn: {burn['dv_kms']:.2f} km/s<extra></extra>",
                showlegend=False,
            ))

    # ── Layout ────────────────────────────────────────────────────────────────
    # Axis range: show full Earth-Moon distance
    ax_range = [-MU * DU_KM * 1.1, (1 - MU) * DU_KM * 1.1]

    fig.update_layout(
        title=dict(text=f"Sample trajectories — {destination_label}", x=0.5,
                   font=dict(size=13, color="#ccc")),
        scene=dict(
            xaxis=dict(title="X (km)", backgroundcolor="#111",
                       gridcolor="#333", color="#aaa", range=ax_range),
            yaxis=dict(title="Y (km)", backgroundcolor="#111",
                       gridcolor="#333", color="#aaa",
                       range=[-DU_KM*0.3, DU_KM*0.3]),
            zaxis=dict(title="Z (km)", backgroundcolor="#111",
                       gridcolor="#333", color="#aaa",
                       range=[-DU_KM*0.3, DU_KM*0.3]),
            bgcolor="#111",
            aspectmode="manual",
            aspectratio=dict(x=3.0, y=0.9, z=0.9),
            camera=dict(eye=dict(x=0.8, y=-1.5, z=0.6)),
        ),
        paper_bgcolor="#111",
        margin=dict(l=0, r=0, t=40, b=0),
        height=460,
    )

    return fig


def build_empty_trajectory_view(message="Trajectories will appear here"):
    """Placeholder 3-D view before computation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#aaa"),
    )
    fig.update_layout(
        scene=dict(bgcolor="#111",
                   xaxis=dict(backgroundcolor="#111", color="#333",
                              showticklabels=False, gridcolor="#222"),
                   yaxis=dict(backgroundcolor="#111", color="#333",
                              showticklabels=False, gridcolor="#222"),
                   zaxis=dict(backgroundcolor="#111", color="#333",
                              showticklabels=False, gridcolor="#222")),
        paper_bgcolor="#111",
        margin=dict(l=0, r=0, t=40, b=0),
        height=460,
    )
    return fig
