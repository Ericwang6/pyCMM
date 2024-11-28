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
        self.bond_indices = torch.tensor(bonds, dtype=torch.long)
        self.angles = []
        self.dihedrals = []
        self.improper_dihedrals = []
        self._find_angles_dihedrals_and_coupling_indices()

    def _find_angles_dihedrals_and_coupling_indices(self):
        """
        Finds all angles and creates the appropriate index tensor.
        Additionally finds all bond-bond and bond-angle couplings
        and creates the appropriate index tensors into self.bond_indices
        and self.angle_indices.

        The bonds array which is passed in is ASSUMED to be sorted such that
        the first row is strictly increasing. This needs to be enforced by
        all parsers. Additionally, the bonds passed in will always
        represent the upper triangle of the adjacency matrix.
        i.e. bonds[n][i] < bonds[n][j] for all n up to N_bonds.

        TODO: In the future, this is where we should create self.dihedral_indices
        and the appropriate couplings for that. Whether or not to construct
        the couplings could be a user input. Probably doesn't make a difference
        if we just always do it but that's an empirical question.
        """
        #where_all = lambda x: torch.where(self.bond_indices[0] == x)
        #maybe = torch.vmap(where_all)(torch.arange(self.bond_indices[0].size(0)))
        
        # @SPEED: I don't know how to do this with magic pytorch functions
        # so this is probably really slow.
        self.bond_bond_indices = torch.empty((0, 2), dtype=torch.long)
        for i in torch.arange(self.bond_indices[0].size(0)):
            # Below excludes indices that only appear once
            bonds_beginning_at_i = torch.where(self.bond_indices[0] == i)[0]
            if bonds_beginning_at_i.size(0) > 1:
                self.bond_bond_indices = torch.cat((self.bond_bond_indices, bonds_beginning_at_i.unsqueeze(0)))
        self.bond_bond_indices = self.bond_bond_indices.T
        #print(self.bond_bond_indices)

        # TODO: I am not completely sure if this correct. We will have to test for other molecules.
        # For instance, when there are centers with three or four bonds, this might not work?
        # I think it will but I'm not completely sure.
        self.angle_indices = torch.stack((
            self.bond_indices[1][self.bond_bond_indices[0]],
            self.bond_indices[0][self.bond_bond_indices[1]],
            self.bond_indices[1][self.bond_bond_indices[1]])
        )
        
        # TODO: Get the bond-angle indices.