import torch
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from .neighbor_list import NeighborList
from .timing_context import TimingContext
from collections import defaultdict, deque

class Topology:
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
        
        # NOTE(JOE): I might revisit the below code and force the intramolecular pairs to have a fixed
        # pair index. This saves on assigning parameters since we don't have to search for which pair
        # indices correspond to intramolecular interactions.
        # Assign a pair index to each pair of atoms in the full set of intramolecular pairs
        #n_exclusions = 0
        #for pair in intramolecular_atomic_pairs:
        #    bonded_atoms_to_pair_index[pair] = n_exclusions
        #    n_exclusions += 1
        #assert n_exclusions == len(intramolecular_atomic_pairs)
        #
        ## Form the atomic pair tensors (in the pair index space) #
        #self.bonded_pairs = torch.tensor([bonded_atoms_to_pair_index[bond] for bond in bonded_atoms], dtype=torch.long, device=self.device, requires_grad=False)
        #self.angle_pairs = torch.tensor(
        #    [(bonded_atoms_to_pair_index[bond_pair[0]], bonded_atoms_to_pair_index[bond_pair[1]]) for bond_pair in angle_bonds],
        #    dtype=torch.long, device=self.device, requires_grad=False
        #).t().contiguous()

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
        single_atom_groups = torch.where(~torch.isin(torch.arange(self.natoms, device=self.device), groups.flatten()))[0].unsqueeze_(1)
        
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
            dtype=torch.long, device=self.device
        )
        self.n_pol_groups = self.pol_group_lengths_g.size(0)
        self.pol_group_segment_indices = torch.zeros(self.n_pol_groups + 1, 
            dtype=torch.long, device=self.device
        )
        self.pol_group_segment_indices[1:] = torch.cumsum(self.pol_group_lengths_g, dim=0)