import torch
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .neighbor_list import NeighborList
from .timing_context import TimingContext
from collections import defaultdict, deque

# The purpose of the Topology object is to walk the molecular graph,
# which is provided as user input in the form of connectivity,
# and construct the index arrays needed for evaluating the potential.
# Once these index arrays are constructed, they will be used to
# index the pairs when computing potentials.

# TODO: Probably the simplest way to get everything we want is to do a breadth-first search over
# the topology beginning with each atom. Stop when we reach a depth of 5. We store the topological
# distance of each atom from every other atom. This let's us easily separate everything into
# inter and intramolecular spaces. We can also query these arrays to get the groups, angles, dihedrals, etc.

class Topology2:
    def __init__(self, bonds: NDArray[np.int64], natoms: int, device: torch.DeviceObjType):
        self.device = device
        self.natoms = natoms
        if bonds.shape[0] == 2:
            self.bonds = bonds.T
        else:
            self.bonds = bonds
        if self.bonds.shape[1] != 2:
            raise ValueError("Bonds list is not of shape (2, N_bonds) or (N_bonds, 2).")
        if isinstance(self.bonds, np.ndarray) == False:
            raise TypeError("Pass in the bonds as a numpy array!")
        
        self.all_paths = self.find_all_paths_within_cutoff()
        self._form_intramolecular_atoms_and_pairs_tensors()
        self._find_atoms_for_building_local_axes()
        self._find_polarization_groups_and_scatter_indices()
    
    def _form_intramolecular_atoms_and_pairs_tensors(self):
        bonded_atoms_to_pair_index = {}
        bonded_atoms = []
        angle_atoms = []
        angle_bonds = []
        intramolecular_atomic_pairs = []
        n_bonds = 0
        n_angles = 0
        for i_atom in self.all_paths.keys():
            for i_neighbor in self.all_paths[i_atom].keys(): # A neighbor is anything inside the 1-4 space.
                path = tuple(self.all_paths[i_atom][i_neighbor])
                if len(path) == 2 and path[0] < path[1]:
                    bonded_atoms.append(path)
                    n_bonds += 1
                    
                    intramolecular_atomic_pairs.append(path)
                    intramolecular_atomic_pairs.append((path[1], path[0]))
                elif len(path) == 3 and path[0] < path[2]:
                    angle_atoms.append(path)
                    angle_bonds.append((tuple(sorted((path[0], path[1]))), tuple(sorted((path[2], path[1])))))
                    n_angles += 1

                    intramolecular_atomic_pairs.append((path[2], path[0]))
                    intramolecular_atomic_pairs.append((path[0], path[2]))
                #elif len(path) == 4 and whatever the symmetry conditions for dihedrals are.
        assert n_bonds == self.bonds.shape[0]
        # NOTE(JOE): The above only gets symmetry-distinct bonds (i<j) and angles (i<k for angle i,j,k)
        
        # Form the atomic pair tensors (in the atomic index space) #
        self.bonded_atoms = torch.tensor(bonded_atoms, dtype=torch.long, device=self.device, requires_grad=False).t().contiguous()
        self.angle_atoms = torch.tensor(angle_atoms, dtype=torch.long, device=self.device, requires_grad=False)
        self.all_intramolecular_pairs = torch.tensor(intramolecular_atomic_pairs, dtype=torch.long, device=self.device, requires_grad=False)
        
        # Assign a pair index to each pair of atoms in the full set of intramolecular pairs
        n_exclusions = 0
        for pair in intramolecular_atomic_pairs:
            bonded_atoms_to_pair_index[pair] = n_exclusions
            n_exclusions += 1
        assert n_exclusions == len(intramolecular_atomic_pairs)
        
        # Form the atomic pair tensors (in the pair index space) #
        self.bonded_pairs = torch.tensor([bonded_atoms_to_pair_index[bond] for bond in bonded_atoms], dtype=torch.long, device=self.device, requires_grad=False)
        self.angle_pairs = torch.tensor(
            [(bonded_atoms_to_pair_index[bond_pair[0]], bonded_atoms_to_pair_index[bond_pair[1]]) for bond_pair in angle_bonds],
            dtype=torch.long, device=self.device, requires_grad=False
        ).t().contiguous()

    def build_adjacency_list(self):
        adj_list = defaultdict(list)
        for u, v in self.bonds:
            adj_list[u].append(v)
            adj_list[v].append(u)
        return adj_list

    def bfs_within_cutoff(self, start, cutoff=4):
        """
        Find all vertices within cutoff distance from start vertex
        Returns a dictionary of {vertex: distance}
        """
        queue = deque([(start, 0)])  # (vertex, distance)
        distances = {start: 0}

        while queue:
            vertex, dist = queue.popleft()

            if dist >= cutoff:
                continue

            for neighbor in self.adj_list[vertex]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        # Do not include distance to self #
        distances.pop(start)

        return distances

    def bfs_paths_within_cutoff(self, start, cutoff=4):
        """
        Find all vertices within cutoff distance from start vertex
        Returns a dictionary of {vertex: path}, where path is a list of vertices 
        from start to the destination vertex, excluding paths to the start vertex itself
        """
        queue = deque([(start, [start], 0)])  # (vertex, path_so_far, distance)
        paths = {}
        visited = {start}

        while queue:
            vertex, path, dist = queue.popleft()

            # Store the path for this vertex
            if vertex != start:  # Skip the start vertex
                paths[vertex] = path

            if dist >= cutoff:
                continue

            for neighbor in self.adj_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, dist + 1))

        return paths
    
    def find_all_paths_within_cutoff(self, cutoff=4):
        self.adj_list = self.build_adjacency_list()
        all_vertices = set(self.adj_list.keys())
        all_paths = {}

        for vertex in all_vertices:
            all_paths[vertex] = self.bfs_paths_within_cutoff(vertex, cutoff)

        return all_paths

    def _find_atoms_for_building_local_axes(self):
        self.xatoms = torch.full((self.natoms,), -1, device=self.device)
        self.yatoms = torch.full((self.natoms,), -1, device=self.device)
        self.zatoms = torch.full((self.natoms,), -1, device=self.device)
        if self.bonded_atoms.size(0) == 0:
            return
        
        # This is basically a way of finding all the 1-2 and 1-3 atoms I think.
        for i in torch.arange(self.natoms):
            all_bonded_atoms_i = torch.concat((
                self.bonded_atoms[1, (self.bonded_atoms[0] == i)], self.bonded_atoms[0, (self.bonded_atoms[1] == i)]
            ))
            if all_bonded_atoms_i.size(0) == 1:
                self.zatoms[i] = all_bonded_atoms_i[0]
            elif all_bonded_atoms_i.size(0) == 2:
                self.zatoms[i] = all_bonded_atoms_i[0]
                self.xatoms[i] = all_bonded_atoms_i[1]
                self.xatoms[all_bonded_atoms_i[0]] = all_bonded_atoms_i[1]
                self.xatoms[all_bonded_atoms_i[1]] = all_bonded_atoms_i[0]
            # TODO: Do the 3-atom and higher cases here once we have examples of that.

    def _find_polarization_groups_and_scatter_indices(self):
        # I think in the general case, this will be the full 1-4 (or maybe 1-3) space of each atom.
        # So, the polarization groups will be OVERLAPPING unlike in AMOEBA.
        
        # TODO: This also only works for water and ions because water's angle atoms
        # are the polarization group and I already collapsed them into one set.
        # Normally, we would have to eliminate all identical sets of atoms so that
        # we do not have duplicate polarization groups.
        if self.angle_atoms.size(0) > 0:
            groups = torch.sort(self.angle_atoms, dim=1).values
        else:
            groups = torch.empty_like(self.angle_atoms)
        single_atom_groups = torch.where(~torch.isin(torch.arange(self.natoms, device=self.bonded_pairs.device), groups.flatten()))[0].unsqueeze_(1)
        
        if single_atom_groups.numel() > 0:
            self.polarization_groups = torch.nested.nested_tensor(list(groups.unbind() + single_atom_groups.unbind()), device=self.device, requires_grad=False)
        else:
            self.polarization_groups = torch.nested.nested_tensor(list(groups.unbind()), device=self.device, requires_grad=False)
        
        # NOTE(JOE): The below code should still work for scattering between groups and atoms
        # even when the groups are overlapping. The code above this comment is not general.
        # The point is, only the code for finding the groups needs to be changed. The below
        # should just work...
        self.pol_group_indices_a = torch.cat(self.polarization_groups.unbind())
        self.pol_group_lengths_g = torch.tensor(
            [g.size(0) for g in self.polarization_groups.unbind()], 
            dtype=torch.long, device=groups.device
        )
        self.n_pol_groups = self.pol_group_lengths_g.size(0)
        self.pol_group_segment_indices = torch.zeros(self.n_pol_groups + 1, 
            dtype=torch.long, device=groups.device
        )
        self.pol_group_segment_indices[1:] = torch.cumsum(self.pol_group_lengths_g, dim=0)

class Topology:
    def __init__(self, bonds: NDArray[np.int64], nl: NeighborList, natoms: int):
        self.device = nl.device
        self.natoms = natoms
        if torch.is_tensor(bonds):
            self.bonded_atoms = bonds.clone().detach()
        else:
            self.bonded_atoms = torch.tensor(bonds, dtype=torch.long, device=nl.device, requires_grad=False)
        self.rebuild(nl)
        self._find_atoms_for_building_local_axes()
        self._find_polarization_groups_and_scatter_indices()
        
    def rebuild(self, nl: NeighborList):
        with TimingContext("topology/indices"):
            self.find_bond_and_angle_pair_indices(nl)

    def _find_bond_indices_i(self, i: torch.Tensor, pairs_i: torch.Tensor, n_neighbors: torch.Tensor):
        """
        Finds all bonds beginning with atom i and writes the resulting index tensor as indices into
        the pairs list. We also find all unique angles formed by bonds around center i.
        We need to enforce some kind of sorting on the bonded_atoms which are passed in so that
        we only have to store the unique bond_indices (i<j). Currently nothing actually
        enforces that condition I am pretty sure.
        """
        # HERE: Need to do a rewrite involving the neighbor list and topology to speed everything up.
        with TimingContext("topology/indices/bonded_pairs/bond_indices/impl"):
            if self.bonded_atoms.size(0) == 0:
                return torch.empty_like(self.bonded_atoms), torch.empty((0, 0), dtype=self.bonded_atoms.dtype, device=self.device)

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

    def _find_atoms_for_building_local_axes(self):
        self.xatoms = torch.full((self.natoms,), -1, device=self.device)
        self.yatoms = torch.full((self.natoms,), -1, device=self.device)
        self.zatoms = torch.full((self.natoms,), -1, device=self.device)

        if self.bonded_atoms.size(0) == 0:
            return
        
        # This is basically a way of finding all the 1-2 and 1-3 atoms I think.
        for i in torch.arange(self.natoms):
            all_bonded_atoms_i = torch.concat((
                self.bonded_atoms[1, (self.bonded_atoms[0] == i)], self.bonded_atoms[0, (self.bonded_atoms[1] == i)]
            ))
            if all_bonded_atoms_i.size(0) == 1:
                self.zatoms[i] = all_bonded_atoms_i[0]
            elif all_bonded_atoms_i.size(0) == 2:
                self.zatoms[i] = all_bonded_atoms_i[0]
                self.xatoms[i] = all_bonded_atoms_i[1]
                self.xatoms[all_bonded_atoms_i[0]] = all_bonded_atoms_i[1]
                self.xatoms[all_bonded_atoms_i[1]] = all_bonded_atoms_i[0]
            # TODO: Do the 3-atom and higher cases here once we have examples of that.

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
        if we just always do it.
        """
        self.bonded_pairs = torch.tensor([], dtype=torch.long, device=nl.device)
        self.angle_pairs = torch.tensor([], dtype=torch.long, device=nl.device)
        # @SPEED Avoid the for loop below. I have to use it because the way the NL
        # works, we cannot use vmap. Need to figure out how to fix that.

        # @SPEED Write a different version of this which updates the topology
        # for only those atoms which actually have a change in their neighbors.
        with TimingContext("topology/indices/bonded_pairs"):
            neighbors_per_atom = nl.get_n_neighbors()
            for i in torch.arange(self.natoms):
                with TimingContext("topology/indices/bonded_pairs/bond_indices"):
                    bonded_pairs_i, angle_pairs_i = self._find_bond_indices_i(torch.tensor([i], device=nl.device), nl.get_neighbors(i), neighbors_per_atom) 
                with TimingContext("topology/indices/bonded_pairs/concat"):
                    self.bonded_pairs = torch.concat((self.bonded_pairs, bonded_pairs_i))
                    self.angle_pairs = torch.concat((self.angle_pairs, angle_pairs_i))
        
            # Below specifies which pairs form an angle. This is the same as coupled pairs of bonds.
            # Note that each row of below also pulls out the pairs which are coupled to each angle.
            # Because each pair of bonds automatically forms an angle and each pair in an angle
            # will also be coupled to that angle, we can use self.angle_pairs to compute
            # the angular, bond-bond coupling, and bond-angle coupling potentials.
            self.angle_pairs = self.angle_pairs.t().contiguous()
            pairs = nl.get_pairs()
        
        with TimingContext("topology/indices/intra"):
            self._find_all_intramolecular_pairs(pairs)
        with TimingContext("topology/indices/inter"):
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
        if self.bonded_pairs.size(0) == 0:
            self.angle_atoms = torch.empty_like(self.bonded_atoms)
            self.intramolecular_atom_indices = torch.empty_like(self.bonded_atoms)
            self.all_intramolecular_pairs = torch.empty_like(self.bonded_pairs)
            return

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
        
    def _find_polarization_groups_and_scatter_indices(self):
        # I think in the general case, this will be the full 1-4 (or maybe 1-3) space of each atom.
        # So, the polarization groups will be OVERLAPPING unlike in AMOEBA.
        
        # TODO: This also only works for water and ions because water's angle atoms
        # are the polarization group and I already collapsed them into one set.
        # Normally, we would have to eliminate all identical sets of atoms so that
        # we do not have duplicate polarization groups.
        if self.angle_atoms.size(0) > 0:
            groups = torch.sort(self.angle_atoms, dim=1).values
        else:
            groups = torch.empty_like(self.angle_atoms)
        single_atom_groups = torch.where(~torch.isin(torch.arange(self.natoms, device=self.bonded_pairs.device), groups.flatten()))[0].unsqueeze_(1)
        
        if single_atom_groups.numel() > 0:
            self.polarization_groups = torch.nested.nested_tensor(list(groups.unbind() + single_atom_groups.unbind()), device=self.device, requires_grad=False)
        else:
            self.polarization_groups = torch.nested.nested_tensor(list(groups.unbind()), device=self.device, requires_grad=False)
        
        # NOTE(JOE): The below code should still work for scattering between groups and atoms
        # even when the groups are overlapping. The code above this comment is not general.
        # The point is, only the code for finding the groups needs to be changed. The below
        # should just work...
        self.pol_group_indices_a = torch.cat(self.polarization_groups.unbind())
        self.pol_group_lengths_g = torch.tensor(
            [g.size(0) for g in self.polarization_groups.unbind()], 
            dtype=torch.long, device=groups.device
        )
        self.n_pol_groups = self.pol_group_lengths_g.size(0)
        self.pol_group_segment_indices = torch.zeros(self.n_pol_groups + 1, 
            dtype=torch.long, device=groups.device
        )
        self.pol_group_segment_indices[1:] = torch.cumsum(self.pol_group_lengths_g, dim=0)