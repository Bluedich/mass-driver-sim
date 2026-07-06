"""
Equal-area sphere tiling for Moon launch sites.

A Fibonacci golden-spiral lattice of ``N`` points with spherical-Voronoi cells.
Every tile is one launch site at its centre ``(lat, lon)``; tiles have
near-equal surface area (±~3 %) and mostly-hexagonal shape.  Density is a single
free parameter ``N`` (any tile count).

The tiling is a *deterministic* function of ``N`` — a given ``N`` always
reproduces the identical set of tiles — so results cache correctly across
sessions without storing the geometry.

Coordinate convention matches ``physics.coordinates._surface_unit_vectors``
(selenographic: lon 0 = sub-Earth near side, +East, +North), so tile
centres/polygons line up with the physics and the future 3-D globe view.
"""

import numpy as np
from scipy.spatial import SphericalVoronoi, cKDTree

# Minimum for a non-degenerate spherical Voronoi diagram.
MIN_TILES = 12

_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def latlon_to_xyz(lat_deg, lon_deg):
    """Selenographic (lat, lon)° → unit vector(s); matches _surface_unit_vectors' r_hat.

    Accepts scalars or arrays; returns shape ``(..., 3)``.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    x = -np.cos(lat) * np.cos(lon)
    y =  np.cos(lat) * np.sin(lon)
    z =  np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def xyz_to_latlon(xyz):
    """Unit vector(s) → (lat, lon)° in the selenographic convention (inverse of latlon_to_xyz)."""
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, -x))
    return lat, lon


def split_antimeridian(lat, lon):
    """Insert NaN breaks where a polyline crosses the ±180° meridian.

    Returns ``(x_lat, x_lon)`` arrays suitable for a single Plotly line trace,
    so a cell straddling the dateline draws as two pieces instead of a streak
    spanning the whole map.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    out_lat, out_lon = [lat[0]], [lon[0]]
    for k in range(1, len(lon)):
        if abs(lon[k] - lon[k - 1]) > 180.0:
            out_lat.append(np.nan)
            out_lon.append(np.nan)
        out_lat.append(lat[k])
        out_lon.append(lon[k])
    return np.array(out_lat), np.array(out_lon)


def _fibonacci_sphere(n):
    """``n`` equal-area points on the unit sphere via the golden-spiral lattice."""
    i = np.arange(n, dtype=float)
    z = 1.0 - (2.0 * i + 1.0) / n          # equal-area in z, centred in each band
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = _GOLDEN_ANGLE * i
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1)


class Tiling:
    """An equal-area tiling of the sphere. Fields are parallel, indexed by tile.

    Attributes
    ----------
    n : int
    centers_latlon : (N, 2) array   — (lat, lon)° of each tile centre / launch site
    centers_xyz    : (N, 3) array   — unit vectors of each tile centre
    polygons_latlon: list of (M_k, 2) arrays — closed cell rings in (lat, lon)°
    polygons_xyz   : list of (M_k, 3) arrays — closed cell rings as unit vectors (for 3-D)
    areas          : (N,) array     — solid angle per tile (steradians; sums to 4π)
    """

    def __init__(self, n, centers_latlon, centers_xyz,
                 polygons_latlon, polygons_xyz, areas):
        self.n = int(n)
        self.centers_latlon = centers_latlon
        self.centers_xyz = centers_xyz
        self.polygons_latlon = polygons_latlon
        self.polygons_xyz = polygons_xyz
        self.areas = areas
        self._tree = None

    @property
    def tree(self):
        """Lazily-built KD-tree over centre unit vectors (nearest Euclidean == nearest angle)."""
        if self._tree is None:
            self._tree = cKDTree(self.centers_xyz)
        return self._tree

    def nearest_index(self, xyz):
        """Nearest tile index for each unit vector in ``xyz`` (shape (..., 3) → (...,))."""
        _, idx = self.tree.query(np.asarray(xyz, dtype=float))
        return idx

    def nearest_tile(self, lat_deg, lon_deg):
        """Index of the tile whose centre is nearest the given (lat, lon)°."""
        return int(self.nearest_index(latlon_to_xyz(lat_deg, lon_deg)))

    @property
    def sites(self):
        """List of (lat, lon) launch sites, one per tile — feeds the optimizer."""
        return [(float(la), float(lo)) for la, lo in self.centers_latlon]


def generate_tiling(n_tiles, seed=0):
    """Build a deterministic equal-area ``Tiling`` of ``n_tiles`` cells.

    ``seed`` is accepted for API symmetry but unused — the lattice is fully
    determined by ``n_tiles``.
    """
    n = max(MIN_TILES, int(n_tiles))

    pts = _fibonacci_sphere(n)
    sv = SphericalVoronoi(pts, radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()

    centers_xyz = pts
    lat, lon = xyz_to_latlon(pts)
    centers_latlon = np.stack([lat, lon], axis=-1)
    areas = sv.calculate_areas()

    polygons_xyz, polygons_latlon = [], []
    for region in sv.regions:
        ring = sv.vertices[region]
        ring = np.vstack([ring, ring[0]])            # close the ring
        polygons_xyz.append(ring)
        rlat, rlon = xyz_to_latlon(ring)
        polygons_latlon.append(np.stack([rlat, rlon], axis=-1))

    return Tiling(n, centers_latlon, centers_xyz,
                  polygons_latlon, polygons_xyz, areas)
