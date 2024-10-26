import torch
from abc import ABC, abstractmethod

"""
This file defines the interface and concrete types of NeighborList.
Any NeighborList should define three methods.
    _build: Build the neighbor list from scratch given positions and a box.
    update: Update the neighbor list without a full rebuild if possible. Otherwise call _build.
    get_neighbors: Return a torch.Tensor of integers specifying all neighbors with the cutoff
    for a particular atom index.

Features that are not currently supported but should be in the future:
1) Non-orthorhombic boxes
2) Half neighbor lists. That is, ones where only unique pairs are stored (i<j).
3) Filtering of neighbors using a masked list. Usually this will mean filtering
out neighbors within a certain a number of bonds which will be much easier to
implement once the topology object is sorted out.
"""

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

class NSquaredList(NeighborList):
    def __init__(self, positions: torch.Tensor, box_lengths: torch.Tensor, cutoff: float):
        """
        Initialize NSquaredList structure which computes all pairwise distances,
        respecting PBCs. This gives the exact neighbor list without any cutoff.
        It should only be used for very small systems and for testing that other
        neighbor lists are being constructed correctly.

        Args:
            positions (torch.Tensor): (N, 3) array of atomic positions
            box_lengths (torch.Tensor): (3,) array of periodic box lengths
        """
        self.cutoff = cutoff
        self.box_lengths = box_lengths
        self.natoms = positions.shape[0]
        self.neighbor_list = torch.full((self.natoms, self.natoms), -1, dtype=torch.long, device=positions.device)
        self.n_neighbors = torch.zeros(self.natoms, dtype=torch.long, device=positions.device)

        self._build(positions)

    def _build(self, positions: torch.Tensor):
        """
        Calculate the distance matrix between atoms with periodic boundary conditions
        for an orthorhombic cell, following the minimum image convention.

        Args:
            positions (torch.Tensor): Nx3 tensor of atomic positions

        Returns:
            torch.Tensor: NxN tensor of minimum image distances
        """
        # Reset neighbor counts
        self.n_neighbors.zero_()
        self.neighbor_list.fill_(-1)

        # Reshape positions for broadcasting
        pos_i = positions.view(self.natoms, 1, 3)  # Shape: N x 1 x 3
        pos_j = positions.view(1, self.natoms, 3)  # Shape: 1 x N x 3

        # Calculate direct differences
        diff = pos_i - pos_j  # Shape: N x N x 3

        # Apply minimum image convention
        # First wrap differences into range [-L/2, L/2]
        diff = diff - torch.round(diff / self.box_lengths) * self.box_lengths

        # Calculate distances
        distances = torch.sqrt(torch.sum(diff * diff, dim=-1))
        indices = torch.where((distances < self.cutoff) & (distances > 0.0), True, False).nonzero()
        for i in range(self.natoms):
            neighbors_i = indices[indices[:, 0] == i][:, 1]
            self.n_neighbors[i] = neighbors_i.size(0)
            self.neighbor_list[i, :][:self.n_neighbors[i]] = neighbors_i
    
    def update(self, positions: torch.Tensor):
        self._build(positions)

    def get_neighbors(self, atom_idx: int):
        return self.neighbor_list[atom_idx, :self.n_neighbors[atom_idx]]


class CellList(NeighborList):
    def __init__(self, positions: torch.Tensor, box_lengths: torch.Tensor, cutoff: float, max_neighbors: int=2048):
        """
        Initialize cell list structure.
        
        Args:
            positions (torch.Tensor): (N, 3) array of atomic positions
            box_lengths (torch.Tensor): (3,) array of periodic box lengths
            cutoff (float): Interaction cutoff distance
            max_neighbors (int): Maximum number of neighbors per atom
        """
        self.minimum_vector, _ = torch.min(positions, dim=0)
        self.cutoff = cutoff
        self.box_lengths = box_lengths
        self.max_neighbors = max_neighbors
        self.num_updates_since_last_build = 0
        self.last_positions = positions.detach().clone()
        
        # Compute cell grid dimensions
        if cutoff > min(box_lengths):
            assert False, "You requested a cutoff that is larger than the smallest box direction. We can't handle this currently. Set the cutoff to the smallest box direction or smaller."
        
        # Find number of cells in each direction then compute all valid cells.
        self.n_cells = torch.floor(box_lengths / cutoff).long()
        self.cell_size = box_lengths / self.n_cells
        
        # Initialize cell assignments
        self.n_atoms = positions.shape[0]
        
        # Initialize neighbor list storage
        self.neighbor_list = torch.full((self.n_atoms, max_neighbors), -1, 
                                      dtype=torch.long, device=positions.device)
        self.n_neighbors = torch.zeros(self.n_atoms, dtype=torch.long, 
                                     device=positions.device)
        
        # Build cell structure
        self._build(positions)
    
    def _build(self, positions: torch.Tensor):
        """
        Build cell list structure from scratch.
        
        Args:
            positions (torch.Tensor): (N, 3) array of atomic positions
        """
        # Convert positions to cell indices
        self._positions_to_cell_indices(positions - self.minimum_vector)
        
        # Update the fixed-size neighbor lists
        self._update_neighbor_lists(positions)
    
    def _update_neighbor_lists(self, positions: torch.Tensor):
        """
        Update the fixed-size neighbor lists for all atoms.
        
        Args:
            positions (torch.Tensor): (N, 3) array of atomic positions
        """
        # Reset neighbor counts
        self.n_neighbors.zero_()
        self.neighbor_list.fill_(-1)
        
        # Update neighbors for each atom
        for i in range(self.n_atoms):
            # Get potential neighbors
            neighbors = self._get_cell_neighbors(i)
            if len(neighbors) == 0:
                continue
                
            # Calculate distances to all potential neighbors
            pos_i = positions[i]
            pos_j = positions[neighbors]
            
            # Apply minimum image convention
            dr = pos_j - pos_i
            dr = dr - torch.round(dr / self.box_lengths) * self.box_lengths
            dist2 = torch.sum(dr * dr, dim=1)
            
            # Select neighbors within cutoff ignoring atoms on top of this atom (likely the atom itself).
            mask = (dist2 < self.cutoff * self.cutoff) & (dist2 > 0)
            valid_neighbors = neighbors[mask]
            
            # Store up to max_neighbors closest neighbors
            n_valid = min(len(valid_neighbors), self.max_neighbors)
            if n_valid > 0:
                self.n_neighbors[i] = n_valid
                self.neighbor_list[i, :n_valid] = valid_neighbors[:n_valid]
    
    def _get_cell_neighbors(self, atom_idx: int):
        """
        Get potential neighbors from neighboring cells for a given atom.
        
        Args:
            atom_idx (int): Index of atom to get neighbors for
            
        Returns:
            torch.Tensor: Array of neighbor indices
        """
        cell_i = self.cell_indices[atom_idx]
        
        cell_x = cell_i[0]
        cell_y = cell_i[1]
        cell_z = cell_i[2]

        # Get neighboring cells (including periodic images)
        # @SPEED: Can vectorize this in some way probably.
        # This also uses the CPU since we accumulate into the
        # neighbor cells below. Vectorizing would allow this to
        # all occur on the GPU.
        neighbor_cells = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    # Apply periodic boundary conditions to cell indices
                    nx = (cell_x + dx) % self.n_cells[0]
                    ny = (cell_y + dy) % self.n_cells[1]
                    nz = (cell_z + dz) % self.n_cells[2]
                    test_cell = torch.Tensor([nx, ny, nz]).expand(self.cell_indices.size(0), 3)

                    neighbor_indices = torch.nonzero(torch.all(torch.eq(test_cell, self.cell_indices), 1)).flatten()
                    if neighbor_indices.numel() != 0:
                        neighbor_cells.append(neighbor_indices)
        
        # Combine all neighbors and remove the atom itself
        if neighbor_cells:
            neighbors = torch.cat(neighbor_cells).unique()
            neighbors = neighbors[neighbors != atom_idx]
            return neighbors
        
        return torch.tensor([], dtype=torch.long)
    
    def _positions_to_cell_indices(self, positions: torch.Tensor) -> None:
        """
        Figure out which cells each atom belongs to and store the result.
        We then store the atom indices which are in each cell.
        Note that the positions
        are shifted so that all atoms lie at positive positions which simplifies
        this calculation. Stores the result in self.cell_indices
        """
        scaled_coords = (positions / self.cell_size).long()
        self.cell_indices = scaled_coords % self.n_cells
    
    def _needs_rebuild(self, positions: torch.Tensor):
        # Find max displacement along a particular axis and see if it exceeds
        # half the smallest cell size. If so, we have to rebuild. We could
        # technically do this matching the cell directions and rebuild less often.
        max_displacement = torch.max(torch.abs(positions - self.last_positions))
        return max_displacement > 0.5 * torch.min(self.cell_size)

    def update(self, positions: torch.Tensor) -> None:
        """
        Update cell list with new positions if needed.
        Increments a counter that keeps track of how many updates
        succeeded without requiring a rebuild. Rebuilds are only
        needed when an atom moves more than half the cell length.
        
        Args:
            positions (torch.Tensor): (N, 3) array of new positions
        """
        # Check if a rebuild of the cells is needed
        if self._needs_rebuild(positions):
            self._build(positions)
            self.last_positions = positions.detach().clone()
            self.num_updates_since_last_build = 0
            return
        
        # Update cell assignments
        self._positions_to_cell_indices(positions)
        
        # Update neighbor lists with new positions
        self._update_neighbor_lists(positions)
        
        self.last_positions = positions.detach().clone()
        self.num_updates_since_last_build += 1
        return
    
    def get_neighbors(self, atom_idx: int):
        """
        Get neighbors for a given atom from the pre-computed neighbor list.
        
        Args:
            atom_idx (int): Index of atom to get neighbors for
            
        Returns:
            torch.Tensor: Array of neighbor indices (padded with -1)
        """
        return self.neighbor_list[atom_idx, :self.n_neighbors[atom_idx]]

if __name__ == "__main__":
    grid = torch.linspace(-10.0, 10.0, 10)
    positions = torch.cartesian_prod(grid, grid, grid)
    box_lengths = torch.tensor([20.1, 20.1, 20.1])
    cutoff = 6.0
    
    # Create cell list
    cell_list = CellList(positions, box_lengths, cutoff)

    # Create NSquareList
    nsq_list = NSquaredList(positions, box_lengths, cutoff)
    print(cell_list.n_neighbors)
    print(cell_list.get_neighbors(0, True))
    