import torch
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

# The purpose of the Topology object is to walk the molecular graph,
# which is provided as user input in the form of connectivity,
# and construct the index arrays needed for evaluating the potential.
# Once these index arrays are constructed, they will be used to
# build the coordinate manager and the parameters object.

class Topology:
    def __init__(self, bonds: NDArray[np.int64]):
        # NOTE: The bond indices are stored in this reason just because
        # that was the original choice. It doesn't make a difference
        # but we are staying with this for convenience.
        self.bond_indices = torch.tensor(bonds, dtype=torch.long)
        self.angles = []
        self.dihedrals = []
        self.improper_dihedrals = []

    def find_angles_and_coupling_indices(self):
        pass