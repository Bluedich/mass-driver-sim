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
            with keys 'x','y','z','dv_kms').  A burn may additionally carry an
            'orbit_id' naming which target_orbits() entry it inserts into.
        """

    def target_orbits(self):
        """
        Insertion orbit(s) for this destination as closed 3-D curves in the
        CR3BP rotating frame (non-dim DU), for visualisation.  Independent of
        any particular trajectory.

        Returns
        -------
        list of dict, each with keys:
            'id'    : str        stable id; matched against burn['orbit_id']
            'label' : str        shown on hover
            'x','y','z' : 1-D arrays (non-dim DU)
        Empty list if the destination exposes no displayable orbit.
        """
        return []
