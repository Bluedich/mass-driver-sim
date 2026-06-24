"""Abstract base class for launch destinations."""

from abc import ABC, abstractmethod
import numpy as np


class Destination(ABC):
    """
    A destination is responsible for:
      1. Computing the post-launch ΔV for a given initial CR3BP state.
      2. Building the suitability grid over the Moon surface.
      3. Providing sample trajectories for visualisation.
    """

    id: str        # unique identifier, used as cache key
    label: str     # human-readable name shown in the UI
    default_insertion_mode: str = "both"   # subclasses override as needed

    @abstractmethod
    def compute_deltav(self, state0):
        """
        Propagate from state0 and return (dv_nondim, trajectory_dict).

        Parameters
        ----------
        state0 : array-like shape (6,)
            CR3BP state at launch (non-dimensional).

        Returns
        -------
        dv : float
            Total post-launch ΔV in non-dimensional units (DU/TU).
            Return np.inf if destination is unreachable.
        trajectory : dict or None
            Keys: 't', 'x', 'y', 'z' (arrays, non-dim), 'burns' (list of dicts
            with keys 'x','y','z','dv_kms').
        """

    def compute_grid(self, lats, lons, progress_cb=None, n_workers=4):
        """
        Compute suitability ΔV grid over (lats × lons).

        Parameters
        ----------
        lats : 1-D array (degrees)
        lons : 1-D array (degrees)
        progress_cb : callable(fraction) or None
        n_workers : int, parallel workers

        Returns
        -------
        dv_grid : 2-D array (len(lats), len(lons)), ΔV in km/s
        sample_trajectories : list of trajectory dicts
        """
        from concurrent.futures import ProcessPoolExecutor
        from physics.optimizer import compute_grid

        pairs = [(lat, lon) for lat in lats for lon in lons]
        total = len(pairs)

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                dv_grid, trajs = compute_grid(lats, lons, self, pool=pool)
        else:
            dv_grid, trajs = compute_grid(lats, lons, self, pool=None)

        if progress_cb:
            progress_cb(1.0)

        return dv_grid, trajs
