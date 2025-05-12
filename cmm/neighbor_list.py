import torch
from abc import ABC, abstractmethod
from typing import Optional
from .pbc import applyPBC
from .timing_context import TimingContext

__all__ = ['NeighborList', 'NSquaredList', 'VerletList']

class NeighborList(ABC):

    @abstractmethod
    def _build(self, positions: torch.Tensor):
        pass

    @abstractmethod
    def update(self, positions: torch.Tensor):
        pass

    @abstractmethod
    def get_all_pairs(self):
        pass

    @abstractmethod
    def get_excluded_pairs(self):
        pass
    
    @abstractmethod
    def get_included_pairs(self):
        pass

    def get_pair_indices(self, pairs_a: torch.Tensor):
        """
        Find indices by hashing each pair of integers and looking up the right index
        using a hash table that is re-computed each time the neighbor list is updated.

        Args:
            pairs_a: Tensor of shape (N, 2) containing integer pairs to find

        Returns:
            indices: Tensor of shape (N,) containing indices where each pair appears in self.all_pairs
        """
        pairs_a_hashes = pairs_a[:, 0] * self.natoms + pairs_a[:, 1]
        pair_indices = self._hash_indices[pairs_a_hashes]
        return pair_indices
    
    def get_pair_indices_search(self, pairs_a: torch.Tensor):
        return (pairs_a.unsqueeze(1) == self.all_pairs.unsqueeze(0)).all(dim=2).nonzero(as_tuple=True)[1].reshape(-1)

class NSquaredList(NeighborList):
    def __init__(self, positions: torch.Tensor, box: torch.Tensor, cutoff: torch.Tensor, excluded_atomic_pairs: Optional[torch.Tensor]=None):
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
        self.excluded_pairs = excluded_atomic_pairs
        self.included_pairs = None
        self.pairs = None
        
        self._hash_indices = torch.zeros(self.natoms * self.natoms, dtype=torch.long, device=self.device)
        
        self._build(positions, box)

    def _build(self, positions: torch.Tensor, box: torch.Tensor):
        """
        Calculate the distance matrix between atoms with periodic boundary conditions
        for a general cell, following the minimum image convention.
        """

        # Reshape positions for broadcasting
        pos_i = positions.view(self.natoms, 1, 3)  # Shape: N x 1 x 3
        pos_j = positions.view(1, self.natoms, 3)  # Shape: 1 x N x 3

        # Calculate direct differences and apply minimum image convention
        distance_vecs = pos_j - pos_i  # Shape: N x N x 3
        distance_vecs = applyPBC(distance_vecs, box, torch.inverse(box))
        distances = torch.linalg.vector_norm(distance_vecs, dim=-1)
        
        # Find pairs #
        pairs_inside_cutoff = torch.where((distances < self.cutoff) & (distances > 0.0), True, False).nonzero()
        self.all_pairs = torch.concat([pairs_inside_cutoff[pairs_inside_cutoff[:, 0] == i] for i in range(self.natoms)], dim=0)

        # Remove excluded pairs and store result as self.included_pairs #
        if self.excluded_pairs is not None:
            included_pair_indices = (~torch.any(torch.all(self.all_pairs.unsqueeze(0) == self.excluded_pairs.unsqueeze(1), dim=2), dim=0)).nonzero().squeeze_()
            self.included_pairs = self.all_pairs[included_pair_indices]
        else:
            self.included_pairs = self.all_pairs

        # Compute the hash values for each pair #
        all_pairs_hashes = self.all_pairs[:, 0] * self.natoms + self.all_pairs[:, 1]
        self._hash_indices[all_pairs_hashes] = torch.arange(len(self.all_pairs), device=self.device)

    def update(self, positions: torch.Tensor, box: torch.Tensor, cutoff: Optional[torch.Tensor]=None) -> None:
        if cutoff:
            self.cutoff = cutoff
        self._build(positions, box)

    def get_all_pairs(self):
        return self.all_pairs
    
    def get_excluded_pairs(self):
        return self.excluded_pairs
    
    def get_included_pairs(self):
        return self.included_pairs

class VerletList(NeighborList):
    def __init__(self, positions: torch.Tensor, box: torch.Tensor, cutoff: torch.Tensor, excluded_atomic_pairs: Optional[torch.Tensor]=None, padding: torch.Tensor=torch.tensor(1.5)):
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
        self.cutoff = cutoff
        self.excluded_pairs = excluded_atomic_pairs
        self.included_pairs = None
        self.pairs = None
        
        self._reference_positions = positions.clone().detach()
        self._reference_box = box.clone().detach()
        self._cutoff_verlet = cutoff + padding
        self._padding = padding
        self._pairs_verlet = None
        
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

        # Store new reference positions and box
        self._reference_positions = positions.clone().detach()
        self._reference_box = box.clone().detach()

        # Reshape positions for broadcasting
        pos_i = positions.view(self.natoms, 1, 3)  # Shape: N x 1 x 3
        pos_j = positions.view(1, self.natoms, 3)  # Shape: 1 x N x 3
        # Calculate direct differences and apply minimum image convention
        distance_vecs = pos_j - pos_i  # Shape: N x N x 3
        distance_vecs = applyPBC(distance_vecs, box, torch.inverse(box))
        distances = torch.linalg.vector_norm(distance_vecs, dim=-1)
        
        pairs_inside_verlet = torch.where((distances < self._cutoff_verlet) & (distances > 0.0), True, False).nonzero()
        self._verlet_pairs = torch.concat([pairs_inside_verlet[pairs_inside_verlet[:, 0] == i] for i in range(self.natoms)], dim=0)

        self._update_from_verlet_pairs(positions, box)

    def _update_from_verlet_pairs(self, positions: torch.Tensor, box: torch.Tensor):
        # If we are here, it means that we have all of the needed pairs in the list of pairs
        # inside the verlet cutoff. We just need to pull the appropriate pairs from there.
        with TimingContext("nl/update/verlet_build/dists"):
            distance_vecs = positions[self._verlet_pairs[:, 1]] - positions[self._verlet_pairs[:, 0]]
            distance_vecs = applyPBC(distance_vecs, box, torch.inverse(box))
            distances = torch.linalg.vector_norm(distance_vecs, dim=1)
        with TimingContext("nl/update/verlet_build/find_pairs"):
            pairs_inside_cutoff = torch.where((distances < self.cutoff) & (distances > 0.0), True, False).nonzero().squeeze_()
            self.all_pairs = self._verlet_pairs[pairs_inside_cutoff]
            
            # Remove excluded pairs and store as self.included_pairs #
            if self.excluded_pairs is not None:
                included_pair_indices = (~torch.any(torch.all(self.all_pairs.unsqueeze(0) == self.excluded_pairs.unsqueeze(1), dim=2), dim=0)).nonzero().squeeze_()
                self.included_pairs = self.all_pairs[included_pair_indices]
            else:
                self.included_pairs = self.all_pairs
            
            # Compute the hash values for each pair #
            all_pairs_hashes = self.all_pairs[:, 0] * self.natoms + self.all_pairs[:, 1]
            self._hash_indices[all_pairs_hashes] = torch.arange(len(self.all_pairs), device=self.device)

    def update(self, positions: torch.Tensor, box: torch.Tensor, cutoff: Optional[torch.Tensor]=None) -> None:
        with TimingContext("nl/update"):
            if self._needs_update(positions, box, cutoff):
                with TimingContext("nl/update/full_build"):
                    self._build(positions, box)
            else:
                with TimingContext("nl/update/verlet_build"):
                    self._update_from_verlet_pairs(positions, box)

    def get_all_pairs(self):
        return self.all_pairs
    
    def get_excluded_pairs(self):
        return self.excluded_pairs
    
    def get_included_pairs(self):
        return self.included_pairs