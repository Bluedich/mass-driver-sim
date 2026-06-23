"""
Coordinate transforms between Moon surface (selenographic) and CR3BP rotating frame.

Selenographic convention
------------------------
lon = 0   : sub-Earth point (near side)
lon = 180 : far side (anti-Earth)
lon positive East, lat positive North.

CR3BP rotating frame
--------------------
Origin at Earth-Moon barycentre.
Earth at (-mu, 0, 0), Moon at (1-mu, 0, 0).
+x: from Earth toward Moon.
+y: direction of Moon's orbital motion (prograde).
+z: orbit-normal (northward).

Frame alignment
---------------
Since the Moon is synchronously locked, its body frame co-rotates with the
CR3BP frame.  The sub-Earth direction (sel lon=0) points TOWARD Earth, which
is in the −x direction from the Moon.  So:

    X_sel (→Earth)  =  −X_rot
    Y_sel (east)    =  +Y_rot
    Z_sel (north)   =  +Z_rot

Surface velocity
----------------
The Moon surface is stationary in the CR3BP rotating frame.  A mass-driver
payload launched with velocity v_driver in the local selenographic frame has
initial CR3BP rotating-frame velocity equal to v_driver expressed in the
rotating frame — no Ω×r correction needed.
"""

import numpy as np
from .cr3bp import MU, R_MOON_DU, VU_KMS


def _surface_unit_vectors(lat_deg, lon_deg):
    """
    Return (r_hat, north_hat, east_hat) all in the CR3BP rotating frame.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # Outward radial (up) — lon=0 points in −x direction (toward Earth)
    r_hat = np.array([
        -np.cos(lat) * np.cos(lon),   # −x for lon=0 → sub-Earth ✓
         np.cos(lat) * np.sin(lon),   # +y for lon=90 → orbital east ✓
         np.sin(lat),                 # +z northward ✓
    ])

    # North: derivative of r_hat w.r.t. lat (unit length)
    north_hat = np.array([
         np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
         np.cos(lat),
    ])

    # East: derivative of r_hat w.r.t. lon / cos(lat) (unit length)
    east_hat = np.array([
        np.sin(lon),
        np.cos(lon),
        0.0,
    ])

    return r_hat, north_hat, east_hat


def surface_position(lat_deg, lon_deg):
    """
    CR3BP rotating-frame position of a point on the Moon's surface.

    Parameters
    ----------
    lat_deg : geodetic latitude (degrees, −90 to 90)
    lon_deg : selenographic longitude (degrees; 0 = sub-Earth near side)

    Returns
    -------
    pos : ndarray shape (3,), non-dimensional DU
    """
    r_hat, _, _ = _surface_unit_vectors(lat_deg, lon_deg)
    moon_centre = np.array([1.0 - MU, 0.0, 0.0])
    return moon_centre + R_MOON_DU * r_hat


def launch_velocity_rotating(lat_deg, lon_deg, azimuth_deg, elevation_deg, speed_du):
    """
    CR3BP rotating-frame velocity for a mass-driver launch.

    The mass driver fires in the local selenographic frame; since the Moon is
    synchronously locked, this is identical to the rotating frame — no extra
    Ω×r correction is needed.

    Parameters
    ----------
    lat_deg, lon_deg : surface site (degrees)
    azimuth_deg : clockwise from North (0 = North, 90 = East)
    elevation_deg : angle above horizontal (0 = tangent, max 5°)
    speed_du : launch speed in DU/TU

    Returns
    -------
    vel : ndarray shape (3,), DU/TU in CR3BP rotating frame
    """
    r_hat, north_hat, east_hat = _surface_unit_vectors(lat_deg, lon_deg)

    az  = np.radians(azimuth_deg)
    el  = np.radians(elevation_deg)

    # Horizontal direction along azimuth
    horiz = np.cos(az) * north_hat + np.sin(az) * east_hat

    # Launch direction: tilted 'el' degrees above horizontal
    launch_dir = np.cos(el) * horiz + np.sin(el) * r_hat
    # (already unit length since horiz and r_hat are orthogonal unit vectors)

    return speed_du * launch_dir


def initial_state(lat_deg, lon_deg, azimuth_deg, elevation_deg, speed_du):
    """
    Full 6-element CR3BP state [x,y,z,vx,vy,vz] for a surface launch.
    """
    pos = surface_position(lat_deg, lon_deg)
    vel = launch_velocity_rotating(lat_deg, lon_deg, azimuth_deg, elevation_deg, speed_du)
    return np.concatenate([pos, vel])


def rotating_to_earth_inertial(state_rot):
    """
    Convert CR3BP rotating-frame state to Earth-centred inertial state (at t=0).

    Returns [xe, ye, ze, vxi, vyi, vzi] in DU, DU/TU.
    The transformation is:
        r_earth  = r_rot − r_earth_in_rot  (shift origin to Earth)
        v_earth  = v_rot + Ω × r_earth     (add frame-rotation contribution)
    where Ω = ẑ (non-dimensional).
    """
    x, y, z, vx, vy, vz = state_rot
    xe = x + MU
    ye = y
    ze = z
    # Ω × r_earth = (0,0,1) × (xe,ye,ze) = (−ye, xe, 0)
    vxi = vx - ye
    vyi = vy + xe
    vzi = vz
    return np.array([xe, ye, ze, vxi, vyi, vzi])


def speed_km_s(speed_du):
    return speed_du * VU_KMS


def speed_du_per_tu(speed_km_s):
    return speed_km_s / VU_KMS
