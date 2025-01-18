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
        self.bonded_atoms = torch.tensor(bonds, dtype=torch.long, device=nl.device)
        self.find_bond_and_angle_pair_indices(nl)

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
        bonded_pairs_i = half_bond_matches_1 + torch.sum(n_neighbors[:i])
        angle_pairs_i = torch.combinations(bonded_pairs_i, r=2)
        
        # Below will get the same bond pairs as above but will find them in the 
        # reverse order. I don't think we need them ever but I'm not sure yet so leaving the comment.
        #half_bond_ending_with_i = self.bonded_atoms[0, (self.bonded_atoms[1] == i)]
        #half_bond_matches_2 = torch.nonzero(torch.sum((half_bond_ending_with_i.unsqueeze(1) - pairs_i) == 0, dim=0)).flatten()
        #bonded_pairs_backward_i = half_bond_matches_2 + torch.sum(n_neighbors[:i])
        
        return bonded_pairs_i, angle_pairs_i

    def find_bond_and_angle_pair_indices(self, nl: NeighborList):
        """
        The bonds array which is stored here is ASSUMED to be sorted such that
        the first row is strictly increasing. This needs to be enforced by
        all parsers. Additionally, the bonds passed in will always
        represent the upper triangle of the adjacency matrix.
        i.e. bonds[n][i] < bonds[n][j] for all n up to N_bonds.

        TODO: In the future, this is where we should create self.dihedral_indices
        and the appropriate couplings for that. Whether or not to construct
        the couplings could be a user input. Probably doesn't make a difference
        if we just always do it but that's an empirical question.
        """
        self.bonded_pairs = torch.tensor([], dtype=torch.long, device=nl.device)
        self.angle_pairs = torch.tensor([], dtype=torch.long, device=nl.device)
        # @SPEED Avoid this for loop. I have to use it because the way the NL
        # works, we cannot use vmap. Need to figure out how to fix that.
        for i in torch.arange(self.natoms):
            bonded_pairs_i, angle_pairs_i = self._find_bond_indices_i(torch.tensor([i], device=nl.device), nl.get_neighbors(i), nl.get_n_neighbors()) 
            self.bonded_pairs = torch.concat((self.bonded_pairs, bonded_pairs_i))
            self.angle_pairs = torch.concat((self.angle_pairs, angle_pairs_i))
        
        # Below specifies which pairs form an angle. This is the same as coupled pairs of bonds.
        # Note that each row of below also pulls out the pairs which are coupled to each angle.
        # Because each pair of bonds automatically forms an angle and each pair in an angle
        # will also be coupled to that angle, we can use self.angle_pairs to compute
        # the angular, bond-bond coupling, and bond-angle coupling potentials.
        self.angle_pairs = self.angle_pairs.t().contiguous()
        
        pairs = nl.get_pairs()
        self._find_all_intramolecular_pairs(pairs)
        self._find_all_intermolecular_pairs(pairs.size(0))

    def _find_all_intermolecular_pairs(self, n_pairs):
        """Compute set difference between intramolecular pairs array and torch.arange over n_pairs."""
        all_pairs = torch.arange(n_pairs, device=self.bonded_pairs.device)
        mask = ~torch.isin(all_pairs, self.all_intramolecular_pairs)
        self.all_intermolecular_pairs = all_pairs[mask]

    def _find_all_intramolecular_pairs(self, pairs: torch.Tensor):
        # Once we have dihedral indices, we should use that instead.
        # Basically, this lets us easily get the indices of all pairs
        # which can be formed from the atoms forming angles/dihedrals.
        # e.g. we need to also capture the H-H pair in a water which
        # is not bonded but should be ignored in the intermolecular
        # calculations.
        # These indexing shenanigans come from: https://stackoverflow.com/questions/73187923/applying-torch-combinations-on-multidimensional-tensor-or-tuple-of-tensors-in-py
        self.angle_atoms = torch.unique(torch.hstack((pairs[self.angle_pairs[0]], pairs[self.angle_pairs[1]])), dim=1)[:, [1, 0, 2]]
        # ^^^ It is unclear to me if the above re-ordering will always work or just for water. These indices are used
        # when evaluating angle-dependent parameters. If that seems to be a problem, then this ordering is probably not
        # guaranteed to be right. The solution is probably some standard sorting of the atoms for how angles are evaluated.
        # i.e. some canonical ordering of the bond graph.
        # -Joe 1/3/25

        c = torch.combinations(torch.arange(self.angle_atoms.size(1), device=pairs.device), r=2)
        x = self.angle_atoms[:,None].expand(-1, c.size(0), -1).to(device=pairs.device)
        idx = c[None].expand(len(x), -1, -1)
        self.intramolecular_atom_indices = x.gather(dim=2, index=idx).reshape(-1, 2)
        mask_1 = torch.where((self.intramolecular_atom_indices == pairs.unsqueeze(1)).all(-1).any(-1))[0]
        mask_2 = torch.where((torch.index_select(self.intramolecular_atom_indices, 1, torch.tensor([1, 0], device=pairs.device)) == pairs.unsqueeze(1)).all(-1).any(-1))[0]
        self.all_intramolecular_pairs, _ = torch.sort(torch.stack((mask_1, mask_2), dim=1).flatten())
        