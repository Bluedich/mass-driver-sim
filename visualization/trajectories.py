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

from physics.cr3bp import MU, R_EARTH_DU, R_MOON_DU, DU_KM, lagrange_points

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


def scene_bounds(trajectories):
    """
    Compute the scene center (cx, cy, cz) and half-span (half) in km from a
    list of trajectory dicts.  Returns floats in km.
    """
    if not trajectories:
        return 0.0, 0.0, 0.0, 1.0
    all_x = np.concatenate([t["x"] for t in trajectories]) * DU_KM
    all_y = np.concatenate([t["y"] for t in trajectories]) * DU_KM
    all_z = np.concatenate([t["z"] for t in trajectories]) * DU_KM
    cx = (all_x.min() + all_x.max()) / 2
    cy = (all_y.min() + all_y.max()) / 2
    cz = (all_z.min() + all_z.max()) / 2
    half = max(all_x.max() - all_x.min(),
               all_y.max() - all_y.min(),
               all_z.max() - all_z.min()) / 2 * 1.15
    return cx, cy, cz, half


def _init_fixed_bounds():
    """Fixed scene bounds spanning L3→L2 in X and L4/L5 in Y, with 15% buffer."""
    lp = lagrange_points()
    xs = [p[0] * DU_KM for p in lp]
    ys = [p[1] * DU_KM for p in lp]
    cx   = (min(xs) + max(xs)) / 2
    cy   = (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * 1.15
    return cx, cy, 0.0, half


_FIXED_BOUNDS = _init_fixed_bounds()


def fixed_scene_bounds():
    """Return (cx, cy, cz, half) in km — fixed across all trajectory views."""
    return _FIXED_BOUNDS


def build_trajectory_view(trajectories, destination_label="", uirevision=None):
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
    ex, ey, ez = _sphere_mesh(-MU, 0, 0, R_EARTH_DU, n=30)
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
    mx, my, mz = _sphere_mesh(1 - MU, 0, 0, R_MOON_DU, n=24)
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

    single = len(trajectories) == 1
    for i, traj in enumerate(trajectories):
        color = "#ffd700" if single else colors[i % len(colors)]
        width = 3 if single else 2
        x_km = traj["x"] * DU_KM
        y_km = traj["y"] * DU_KM
        z_km = traj["z"] * DU_KM

        fig.add_trace(go.Scatter3d(
            x=x_km, y=y_km, z=z_km,
            mode="lines",
            line=dict(color=color, width=width),
            name=f"Trajectory {i+1}",
            hoverinfo="skip",
            showlegend=False,
        ))

        # Launch-site marker: small sphere in world space so it scales with zoom
        if single:
            R_m = 150.0 / DU_KM  # 150 km radius ≈ 1/12 Moon radius
            sx, sy, sz = _sphere_mesh(traj["x"][0], traj["y"][0], traj["z"][0], R_m, n=10)
            fig.add_trace(go.Surface(
                x=sx * DU_KM, y=sy * DU_KM, z=sz * DU_KM,
                colorscale=[[0, "#00ff99"], [1, "#00ff99"]],
                showscale=False, opacity=1.0,
                hoverinfo="skip",
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

    # ── Destination halo orbit ring ───────────────────────────────────────────
    for traj in trajectories:
        if "halo_x" not in traj:
            continue
        hx = np.append(traj["halo_x"], traj["halo_x"][0]) * DU_KM
        hy = np.append(traj["halo_y"], traj["halo_y"][0]) * DU_KM
        hz = np.append(traj["halo_z"], traj["halo_z"][0]) * DU_KM
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.6)", width=1),
            hoverinfo="skip",
            showlegend=False,
        ))
        break   # one halo ring per view

    # ── Lagrange points ───────────────────────────────────────────────────────
    lp_list = lagrange_points()
    lp_x = [p[0] * DU_KM for p in lp_list]
    lp_y = [p[1] * DU_KM for p in lp_list]
    lp_z = [p[2] * DU_KM for p in lp_list]
    lp_labels = [p[3] for p in lp_list]
    fig.add_trace(go.Scatter3d(
        x=lp_x, y=lp_y, z=lp_z,
        mode="markers+text",
        marker=dict(size=5, color="#ff9900", symbol="diamond",
                    line=dict(color="white", width=1)),
        text=lp_labels,
        textposition="top center",
        textfont=dict(color="#ff9900", size=10),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    cx, cy, cz, half = _FIXED_BOUNDS
    x_range = [cx - half, cx + half]
    y_range = [cy - half, cy + half]
    z_range = [cz - half, cz + half]

    fig.update_layout(
        title=dict(text=f"Sample trajectories — {destination_label}", x=0.5,
                   font=dict(size=13, color="#ccc")),
        scene=dict(
            xaxis=dict(title="X (km)", backgroundcolor="#111",
                       gridcolor="#333", color="#aaa", range=x_range),
            yaxis=dict(title="Y (km)", backgroundcolor="#111",
                       gridcolor="#333", color="#aaa", range=y_range),
            zaxis=dict(title="Z (km)", backgroundcolor="#111",
                       gridcolor="#333", color="#aaa", range=z_range),
            bgcolor="#111",
            aspectmode="cube",
            # screen_right = normalize(-eye.y, eye.x, 0).
            # east (+y) on screen-LEFT requires eye.x < 0 (matches moon-map convention);
            # Moon (high +x) right of Earth requires eye.y < 0.
            # eye=(-0.8, -1.2, 0.8): screen_right=(0.83, -0.55, 0) — both satisfied.
            camera=dict(eye=dict(x=-0.8, y=-1.2, z=0.8)),
            uirevision=uirevision,
        ),
        paper_bgcolor="#111",
        margin=dict(l=0, r=0, t=40, b=0),
        height=560,
        uirevision=uirevision,
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
