import torch
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .neighbor_list import NeighborList

# The purpose of the Topology object is to walk the molecular graph,
# which is provided as user input in the form of connectivity,
# and construct the index arrays needed for evaluating the potential.
# Once these index arrays are constructed, they will be used to
# index the pairs when computing potentials.

class Topology:
    def __init__(self, bonds: NDArray[np.int64], nl: NeighborList, natoms: int):
        self.natoms = natoms
        self.bonded_atoms = torch.tensor(bonds, dtype=torch.long)
        self.find_bond_indices(nl)
        
        #self._find_angles_dihedrals_and_coupling_indices()

    def _find_bond_indices_i(self, i: torch.Tensor, pairs_i: torch.Tensor, n_neighbors: torch.Tensor):
        """
        Finds all bonds beginning with atom i and writes the resulting index tensor as indices into
        the pairs list. We also find all unique angles formed by bonds around center i.
        We need to enforce some kind of sorting on the bonded_atoms which are passed in so that
        we only have to store the unique bond_indices (i<j). Currently nothing actually
        enforces that condition I am pretty sure.
        """
        half_bond_starting_with_i = self.bonded_atoms[1, (self.bonded_atoms[0] == i)]
        half_bond_matches_1 = torch.nonzero(torch.sum((half_bond_starting_with_i.unsqueeze(1) - pairs_i) == 0, dim=0)).flatten()
        #half_bond_ending_with_i = self.bonded_atoms[0, (self.bonded_atoms[1] == i)]
        #half_bond_matches_2 = torch.nonzero(torch.sum((half_bond_ending_with_i.unsqueeze(1) - pairs_i) == 0, dim=0)).flatten()
        bonded_pairs_i = half_bond_matches_1 + torch.sum(n_neighbors[:i]) #torch.concat((half_bond_matches_1, half_bond_matches_2)) + torch.sum(n_neighbors[:i])
        angle_pairs_i = torch.combinations(bonded_pairs_i, r=2)
        return bonded_pairs_i, angle_pairs_i

    def find_bond_indices(self, nl: NeighborList):
        self.bonded_pairs = torch.tensor([], dtype=torch.long)
        self.angle_pairs = torch.tensor([], dtype=torch.long)
        # @SPEED Avoid this for loop. I have to use it because the way the NL
        # works, we cannot use vmap. Need to figure out how to fix that.
        for i in torch.arange(self.natoms):
            bonded_pairs_i, angle_pairs_i = self._find_bond_indices_i(torch.tensor([i]), nl.get_neighbors(i), nl.get_n_neighbors()) 
            self.bonded_pairs = torch.concat((self.bonded_pairs, bonded_pairs_i))
            self.angle_pairs = torch.concat((self.angle_pairs, angle_pairs_i))
        
        # Below specifies which pairs form an angle. This is the same as coupled pairs of bonds.
        # Note that each row of below also pulls out the pairs which are coupled to each angle.
        # Because each pair of bonds automatically forms an angle and each pair in an angle
        # will also be coupled to that angle, we can use self.angle_pairs to compute
        # the angular, bond-bond coupling, and bond-angle coupling potentials.
        self.angle_pairs = self.angle_pairs.t().contiguous()

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

        # Every angle is coupled to two bonds by definition.
        # So, to get the bond_angle_indices, we simply take
        # every angle and we interleave the bond-bond indices
        # as those describe the pairs of bonds which form an
        # angle.
        self.bond_angle_indices = torch.stack((torch.stack((
                self.bond_bond_indices[0], self.bond_bond_indices[1]
            ), dim=1).flatten(),
            torch.stack((
                torch.arange(self.angle_indices[1].size(0)), torch.arange(self.angle_indices[1].size(0))
            ), dim=1).flatten())
        )