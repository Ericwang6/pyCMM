import torch
from abc import ABC, abstractmethod
from typing import Optional
from .pbc import applyPBC

__all__ = ['NeighborList', 'NSquaredList', 'VerletList']

class NeighborList(ABC):

    @abstractmethod
    def _build(self, positions: torch.Tensor):
        pass

    @abstractmethod
    def update(self, positions: torch.Tensor):
        pass

    @abstractmethod
    def get_neighbors(self, atom_idx: int):
        pass

    @abstractmethod
    def get_n_neighbors(self):
        pass

    @abstractmethod
    def get_pairs(self):
        pass

class NSquaredList(NeighborList):
    def __init__(self, positions: torch.Tensor, box: torch.Tensor, cutoff: torch.Tensor):
        """
        Initialize NSquaredList structure which computes all pairwise distances,
        respecting PBCs. This gives the exact neighbor list within a cutoff.
        It should only be used for very small systems and for testing that other
        neighbor lists are being constructed correctly.

        Args:
            positions (torch.Tensor): (N, 3) array of atomic positions
            box (torch.Tensor): (3, 3) tensor representing unit cell
            cutoff (torch.Tensor): (1,) cutoff distance
        """
        self.device = positions.device
        self.natoms = positions.shape[0]
        
        self.cutoff = cutoff
        self.neighbor_list = torch.nested.nested_tensor([torch.empty(0, dtype=torch.long, device=self.device) for _ in range(self.natoms)])
        self.n_neighbors = torch.zeros(self.natoms, dtype=torch.long, device=self.device)
        self.pairs = None
        
        self._build(positions, box)

    def _build(self, positions: torch.Tensor, box: torch.Tensor):
        """
        Calculate the distance matrix between atoms with periodic boundary conditions
        for a general cell, following the minimum image convention.
        """
        # Reset neighbor counts
        self.n_neighbors.zero_()

        # Reshape positions for broadcasting
        pos_i = positions.view(self.natoms, 1, 3)  # Shape: N x 1 x 3
        pos_j = positions.view(1, self.natoms, 3)  # Shape: 1 x N x 3
        # Calculate direct differences and apply minimum image convention
        distance_vecs = pos_i - pos_j  # Shape: N x N x 3
        distance_vecs = applyPBC(distance_vecs, box, torch.inverse(box))
        # Calculate distances
        distances = torch.linalg.vector_norm(distance_vecs, dim=-1)
        
        pairs_inside_cutoff = torch.where((distances < self.cutoff) & (distances > 0.0), True, False).nonzero()
        neighbors = [pairs_inside_cutoff[pairs_inside_cutoff[:, 0] == i][:, 1] for i in range(self.natoms)]
        self.neighbor_list = torch.nested.nested_tensor(neighbors)

        for i in range(self.natoms):
            self.n_neighbors[i] = self.neighbor_list.unbind()[i].size(0)
    
        all_pairs_0 = []
        all_pairs_1 = []
        for i in range(self.natoms):
            all_pairs_0.append(torch.full((self.n_neighbors[i],), i, device=self.device),)
            all_pairs_1.append(self.neighbor_list[i])
        self.pairs = torch.stack((torch.cat(all_pairs_0), torch.cat(all_pairs_1)), dim=1)

    def update(self, positions: torch.Tensor, box: torch.Tensor, cutoff: Optional[torch.Tensor]=None) -> None:
        if cutoff:
            self.cutoff = cutoff
        self._build(positions, box)

    def get_neighbors(self, atom_idx: int):
        return self.neighbor_list[atom_idx]
    
    def get_n_neighbors(self):
        return self.n_neighbors

    def get_pairs(self):
        return self.pairs

class VerletList(NeighborList):
    def __init__(self, positions: torch.Tensor, box: torch.Tensor, cutoff: torch.Tensor, padding: torch.Tensor):
        """
        Initialize NSquaredList structure which computes all pairwise distances,
        respecting PBCs. This gives the exact neighbor list within a cutoff.
        It should only be used for very small systems and for testing that other
        neighbor lists are being constructed correctly.

        Args:
            positions (torch.Tensor): (N, 3) array of atomic positions
            box (torch.Tensor): (3, 3) tensor representing unit cell
            cutoff (torch.Tensor): (1,) cutoff distance
            padding (torch.Tensor): (1,) padding added to cutoff to prevent rebuilds
        """
        self.device = positions.device
        self.natoms = positions.shape[0]
        self._reference_positions = positions.clone().detach()
        self._reference_box = box.clone().detach()
        self._cutoff_verlet = cutoff + padding
        self._padding = padding
        self._neighbor_list_verlet = torch.nested.nested_tensor([torch.empty(0, dtype=torch.long, device=self.device) for _ in range(self.natoms)])
        self._n_neighbors_verlet = torch.zeros(self.natoms, dtype=torch.long, device=self.device)
        self._pairs_verlet = None
        
        self.cutoff = cutoff
        self.neighbor_list = torch.nested.nested_tensor([torch.empty(0, dtype=torch.long, device=self.device) for _ in range(self.natoms)])
        self.n_neighbors = torch.zeros(self.natoms, dtype=torch.long, device=self.device)
        self.pairs = None

        self._build(positions, box)
    
    def _needs_update(self, positions: torch.Tensor, box: torch.Tensor, cutoff: Optional[torch.Tensor]=None):
        # If the cutoff changes, always rebuild
        if cutoff is not None and torch.isclose(cutoff, self.cutoff) == False:
            # Update cutoff for NSquaredList
            self._cutoff_verlet = cutoff + self._padding
            self.cutoff = cutoff
            return True
        
        max_displacement =  torch.max(torch.abs(positions - self._reference_positions))
        max_displacement += torch.max(torch.abs(torch.linalg.qr(box - self._reference_box).R)) # Compute maximum change in eigenvalue
        # NOTE(JOE): ^^^ Strcitly speaking, I think this contribution depends on the type of box.
        # I think that for a cubic box, the box sides need to be moved by sqrt(2)/2 = 0.707...
        # in order to guarantee a rebuild. If only one side were shrinking, then the images
        # move half the distance that the side length shrinks
        # (since both sides move toward the center of the box). In that case the rebuild occurs at
        # the verlet cutoff (not half of it). The images in a diagonally connected cell move towards
        # the center of the box faster than that.
        # To be conservative, I just sum these contributions and use half the padding since that
        # is what applies to changes in atomic positions.
        # The point of this comment is to say that we can be more efficient about when the NL
        # needs to be updated with respect to changes in the cell if we want to specialize
        # the rules for the type of box we are using.
        return max_displacement > 0.5 * self._padding
    
    def _build(self, positions: torch.Tensor, box: torch.Tensor):
        """
        Calculate the distance matrix between atoms with periodic boundary conditions
        for a general cell, following the minimum image convention.
        """
        # Reset neighbor counts
        self.n_neighbors.zero_()

        # Store new reference positions and box
        self._reference_positions = positions.clone().detach()
        self._reference_box = box.clone().detach()

        # Reshape positions for broadcasting
        pos_i = positions.view(self.natoms, 1, 3)  # Shape: N x 1 x 3
        pos_j = positions.view(1, self.natoms, 3)  # Shape: 1 x N x 3
        # Calculate direct differences and apply minimum image convention
        distance_vecs = pos_i - pos_j  # Shape: N x N x 3
        distance_vecs = applyPBC(distance_vecs, box, torch.inverse(box))
        # Calculate distances
        distances = torch.linalg.vector_norm(distance_vecs, dim=-1)
        pairs_inside_verlet = torch.where((distances < self._cutoff_verlet) & (distances > 0.0), True, False).nonzero()
        neighbors = [pairs_inside_verlet[pairs_inside_verlet[:, 0] == i][:, 1] for i in range(self.natoms)]
        self._neighbor_list_verlet = torch.nested.nested_tensor(neighbors)

        for i in range(self.natoms):
            self._n_neighbors_verlet[i] = self._neighbor_list_verlet.unbind()[i].size(0)

        all_pairs_0 = []
        all_pairs_1 = []
        for i in range(self.natoms):
            all_pairs_0.append(torch.full((self._n_neighbors_verlet[i],), i, device=self.device),)
            all_pairs_1.append(self._neighbor_list_verlet[i])
        self._verlet_pairs = torch.stack((torch.cat(all_pairs_0), torch.cat(all_pairs_1)), dim=1)

        self._update_from_verlet_pairs(positions, box)

    def _update_from_verlet_pairs(self, positions: torch.Tensor, box: torch.Tensor):
        # If we are here, it means that we have all of the needed pairs in the list of pairs
        # inside the verlet cutoff. We just need to pull the appropriate pairs from there.
        distance_vecs = positions[self._verlet_pairs[:, 1]] - positions[self._verlet_pairs[:, 0]]
        distance_vecs = applyPBC(distance_vecs, box, torch.inverse(box))
        distances = torch.linalg.vector_norm(distance_vecs, dim=1)
        pairs_inside_cutoff = torch.where((distances < self.cutoff) & (distances > 0.0), True, False).nonzero().squeeze_()
        self.pairs = self._verlet_pairs[pairs_inside_cutoff]
        neighbors = [self.pairs[self.pairs[:, 0] == i][:, 1] for i in range(self.natoms)]
        self.neighbor_list = torch.nested.nested_tensor(neighbors)

        for i in range(self.natoms):
            self.n_neighbors[i] = self.neighbor_list.unbind()[i].size(0)

    def update(self, positions: torch.Tensor, box: torch.Tensor, cutoff: Optional[torch.Tensor]=None) -> None:
        if self._needs_update(positions, box, cutoff):
            self._build(positions, box)
        else:
            self._update_from_verlet_pairs(positions, box)

    def get_neighbors(self, atom_idx: int):
        return self.neighbor_list[atom_idx]
    
    def get_n_neighbors(self):
        return self.n_neighbors

    def get_pairs(self):
        return self.pairs