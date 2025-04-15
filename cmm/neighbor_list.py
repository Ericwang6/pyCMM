import torch
from abc import ABC, abstractmethod

__all__ = ['NeighborList', 'NSquaredList', 'CellList', 'VerletList']

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
        self.device = positions.device
        self.cutoff = cutoff
        self.natoms = positions.shape[0]
        self.neighbor_list = torch.full((self.natoms, self.natoms), -1, dtype=torch.long, device=self.device)
        self.n_neighbors = torch.zeros(self.natoms, dtype=torch.long, device=self.device)

        self._build(positions, box_lengths)

    def _build(self, positions: torch.Tensor, box_lengths: torch.Tensor):
        """
        Calculate the distance matrix between atoms with periodic boundary conditions
        for an orthorhombic cell, following the minimum image convention.
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
        diff = diff - torch.round(diff / box_lengths) * box_lengths

        # Calculate distances
        distances = torch.sqrt(torch.sum(diff * diff, dim=-1))
        indices = torch.where((distances < self.cutoff) & (distances > 0.0), True, False).nonzero()
        self.pairs = indices
        for i in range(self.natoms):
            neighbors_i = indices[indices[:, 0] == i][:, 1]
            self.n_neighbors[i] = neighbors_i.size(0)
            self.neighbor_list[i, :][:self.n_neighbors[i]] = neighbors_i
    
    def update(self, positions: torch.Tensor, box_lengths: torch.Tensor) -> None:
        self._build(positions, box_lengths)

    def get_neighbors(self, atom_idx: int):
        return self.neighbor_list[atom_idx, :self.n_neighbors[atom_idx]]
    
    def get_n_neighbors(self):
        return self.n_neighbors

    def get_pairs(self):
        n_pairs = 0
        all_pairs_0 = []
        all_pairs_1 = []
        for i in range(self.natoms):
            all_pairs_0.append(torch.full((self.n_neighbors[i],), i, device=self.device),)
            all_pairs_1.append(self.neighbor_list[i, :self.n_neighbors[i]])
            n_pairs += self.n_neighbors[i]
        return torch.stack((torch.cat(all_pairs_0), torch.cat(all_pairs_1)), dim=1)

class CellList(NeighborList):
    def __init__(self, positions: torch.Tensor, box_lengths: torch.Tensor, cutoff: torch.Tensor, max_neighbors: int=512):
        """
        Initialize cell list structure.
        
        Args:
            positions (torch.Tensor): (N, 3) array of atomic positions
            box_lengths (torch.Tensor): (3,) array of periodic box lengths
            cutoff (float): Interaction cutoff distance
            max_neighbors (int): Maximum number of neighbors per atom
        """
        self.device = positions.device
        self.minimum_vector, _ = torch.min(positions, dim=0)
        self.cutoff = cutoff
        self.box_lengths = box_lengths
        self.max_neighbors = max_neighbors
        self.num_updates_since_last_build = 0
        self.last_positions = positions.detach().clone()
        
        # Compute cell grid dimensions
        if cutoff > 0.5 * min(box_lengths):
            assert False, "You requested a cutoff that is larger than the half the smallest box direction. We can't handle this currently. Set the cutoff to the smallest box direction or smaller."
        
        # Find number of cells in each direction then compute all valid cells.
        self.n_cells = torch.floor(self.box_lengths / self.cutoff).long()
        self.cell_size = self.box_lengths / self.n_cells
        
        # Initialize cell assignments
        self.n_atoms = positions.shape[0]

        # Initialize neighbor list storage
        self.n_pairs = torch.zeros(1, dtype=torch.long, device=self.device)
        self.neighbor_list = torch.full((self.n_atoms, max_neighbors), -1, 
                                      dtype=torch.long, device=self.device)
        
        # @SPEED: I think this might be faster if it were Nx2 rather than 2xN?
        self.pairs = torch.full((2, self.n_atoms * max_neighbors), -1,
                                dtype=torch.long, device=self.device)
        self.n_neighbors = torch.zeros(self.n_atoms, dtype=torch.long, 
                                     device=self.device)
        #self.distance_vectors = torch.zeros((self.n_atoms, max_neighbors, 3),
        #                                    dtype=positions.dtype, device=self.device)
        #self.distances = torch.zeros((self.n_atoms, max_neighbors),
        #                             dtype=positions.dtype, device=self.device)
        
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
        self.n_pairs = 0
        
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
            
            # Select neighbors within cutoff ignoring the atom itself.
            mask = (dist2 < self.cutoff * self.cutoff) & (dist2 > 0)
            valid_neighbors = neighbors[mask]

            # Store up to max_neighbors closest neighbors
            n_valid = min(len(valid_neighbors), self.max_neighbors)
            if n_valid > 0:
                # @SPEED This results in a whole lot of copying.
                self.n_neighbors[i] = n_valid
                self.neighbor_list[i, :n_valid] = valid_neighbors[:n_valid]
                self.pairs[0][self.n_pairs:(self.n_pairs+n_valid)] = torch.full((n_valid,), i)
                self.pairs[1][self.n_pairs:(self.n_pairs+n_valid)] = valid_neighbors
                self.n_pairs = self.n_pairs + n_valid
                #self.distance_vectors[i, :n_valid, :] = dr[:n_valid]
                #self.distances[i, :n_valid] = torch.sqrt(dist2[:n_valid])
            if n_valid >= self.max_neighbors:
                print(f"Warning: Found {n_valid} neighbors for atom {i} and up to {self.max_neighbors} are allowed. Increase the maximum number of neighbors to ensure interactions are not being omitted erroneously!")
    
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
                    test_cell = torch.tensor([nx, ny, nz], device=self.device).expand(self.cell_indices.size(0), 3)

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

    def update(self, positions: torch.Tensor, box_lengths: torch.Tensor) -> None:
        """
        Update cell list with new positions if needed.
        Increments a counter that keeps track of how many updates
        succeeded without requiring a rebuild. Rebuilds are only
        needed when an atom moves more than half the cell length.
        
        Args:
            positions (torch.Tensor): (N, 3) array of new positions
            box_lengths (torch.Tensor): (3) array of box lengths
        """
        # Check if a rebuild of the cells is needed
        if self._needs_rebuild(positions):
            self.box_lengths = box_lengths
            self.minimum_vector = torch.min(positions, dim=0)[0]
            self._build(positions)
            self.last_positions = positions.detach().clone().requires_grad_(False)
            self.num_updates_since_last_build = 0
            return
        
        self.num_updates_since_last_build += 1
        return

    def get_pairs(self) -> torch.Tensor:
        return self.pairs[:, :self.n_pairs].t().contiguous()

    def get_neighbors(self, atom_idx: int):
        """
        Get neighbors for a given atom from the pre-computed neighbor list.
        
        Args:
            atom_idx (int): Index of atom to get neighbors for
        
        Returns:
            torch.Tensor: Array of neighbor indices (padded with -1)
        """
        return self.neighbor_list[atom_idx, :self.n_neighbors[atom_idx]]
    
    def get_n_neighbors(self):
        return self.n_neighbors

class VerletList(NeighborList):
    def __init__(self, positions: torch.Tensor, box_lengths: torch.Tensor, cutoff: float, 
                 cutoff_padding: float = 0.5, max_neighbors: int = 1024):
        """
        Initialize Verlet list structure.
        
        Args:
            positions (torch.Tensor): (N, 3) array of atomic positions
            box_lengths (torch.Tensor): (3,) array of periodic box lengths
            cutoff (float): Interaction cutoff distance
            cutoff_padding (float): Extra padding distance beyond cutoff for Verlet list 
                                   (determines how frequently the list needs rebuilding)
            max_neighbors (int): Maximum number of neighbors per atom
        """
        self.device = positions.device
        self.cutoff = cutoff
        self.cutoff_padding = cutoff_padding
        self.verlet_cutoff = cutoff + cutoff_padding
        self.box_lengths = box_lengths
        self.max_neighbors = max_neighbors
        
        # Initialize tracking variables
        self.last_positions = positions.detach().clone()
        
        # Initialize neighbor list storage
        self.n_atoms = positions.shape[0]
        self.neighbor_list = torch.full((self.n_atoms, max_neighbors), -1, 
                                        dtype=torch.long, device=self.device)
        self.n_neighbors = torch.zeros(self.n_atoms, dtype=torch.long, 
                                     device=self.device)
        #self.distance_vectors = torch.zeros((self.n_atoms, max_neighbors, 3),
        #                                  dtype=positions.dtype, device=self.device)
        #self.distances = torch.zeros((self.n_atoms, max_neighbors),
        #                           dtype=positions.dtype, device=self.device)
        
        self._build(positions)
    
    def _build(self, positions: torch.Tensor):
        """
        Build Verlet list structure from scratch.
        """
        # Reset neighbor counts and stored displacements
        self.n_neighbors.zero_()
        self.neighbor_list.fill_(-1)
        self.last_positions = positions.detach().clone()
        
        self._update_neighbors_n_squared(positions)
        
    def _update_neighbors_n_squared(self, positions: torch.Tensor):
        """
        Update the neighbor lists for all atoms using the N² algorithm.
        This builds the Verlet list from scratch.
        """
        # Get distance vectors respecting PBCs
        pos_i = positions.view(self.n_atoms, 1, 3)  # Shape: N x 1 x 3
        pos_j = positions.view(1, self.n_atoms, 3)  # Shape: 1 x N x 3
        diff = pos_i - pos_j  # Shape: N x N x 3
        diff = diff - torch.round(diff / self.box_lengths) * self.box_lengths

        # Calculate squared distances (avoid sqrt for filtering)
        dist2 = torch.sum(diff * diff, dim=-1)  # Shape: N x N
        verlet_cutoff2 = self.verlet_cutoff * self.verlet_cutoff
        
        # For each atom i, find all neighbors within the Verlet cutoff
        for i in range(self.n_atoms):
            # Get potential neighbors (excluding self)
            mask = (dist2[i] < verlet_cutoff2) & (dist2[i] > 0.0)
            neighbors = torch.where(mask)[0]
            
            # Store up to max_neighbors neighbors
            n_valid = min(len(neighbors), self.max_neighbors)
            if n_valid > 0:
                self.n_neighbors[i] = n_valid
                self.neighbor_list[i, :n_valid] = neighbors[:n_valid]
                
                # Calculate and store distance vectors and magnitudes
                dr = diff[i][neighbors[:n_valid]]
                self.distance_vectors[i, :n_valid, :] = dr
                self.distances[i, :n_valid] = torch.sqrt(dist2[i][neighbors[:n_valid]])

    def _needs_rebuild(self, positions: torch.Tensor):
        """
        Check if the Verlet list needs to be rebuilt based on atom displacements.
        """
        # Update maximum displacement since last rebuild
        displacements = positions - self.last_positions
        max_displacement = torch.max(torch.norm(displacements, dim=1))
        return max_displacement > (self.cutoff_padding / 2.0)

    def update(self, positions: torch.Tensor, box_lengths: torch.Tensor = None):
        """
        Update Verlet list with new positions if needed.
        Only rebuilds the list if atoms have moved enough.
        """
        # @SPEED We can do better than this but this will have to do for now.
        # Always rebuild if box changes
        if box_lengths is not None and torch.allclose(box_lengths, self.box_lengths) == False:
            self.box_lengths = box_lengths.detach.clone()
            self._build(positions)
            return
            
        if self._needs_rebuild(positions):
            self._build(positions)
        #else:
        #    self._update_distances(positions)
    
    def _update_distances(self, positions: torch.Tensor):
        """
        Update distance vectors and magnitudes without rebuilding the neighbor list.
        """

        for i in range(self.n_atoms):
            n_valid = self.n_neighbors[i]
            if n_valid > 0:
                neighbors = self.neighbor_list[i, :n_valid]
                
                # Calculate new displacement vectors
                dr = positions[neighbors] - positions[i].unsqueeze(0)
                
                # Apply minimum image convention
                dr = dr - torch.round(dr / self.box_lengths) * self.box_lengths
                
                # Update distance vectors and magnitudes
                self.distance_vectors[i, :n_valid, :] = dr
                self.distances[i, :n_valid] = torch.norm(dr, dim=1)

    def get_neighbors(self, atom_idx: int):
        """
        Get neighbors for a given atom from the Verlet list.
        """
        n_valid = self.n_neighbors[atom_idx]
        if n_valid == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        # Filter to only include neighbors within the actual cutoff (not Verlet cutoff)
        mask = self.distances[atom_idx, :n_valid] < self.cutoff
        return self.neighbor_list[atom_idx, :n_valid][mask]
    
    def get_pairs(self):
        """
        Get all unique atom pairs within the cutoff distance.
        """
        pairs = []
        
        # For each atom, get all neighbors within the actual cutoff (not Verlet cutoff)
        for i in range(self.n_atoms):
            n_valid = self.n_neighbors[i]
            if n_valid == 0:
                continue
                
            # Apply the actual cutoff (not the Verlet cutoff with padding)
            mask = self.distances[i, :n_valid] < self.cutoff
            neighbors = self.neighbor_list[i, :n_valid][mask]
            
            if len(neighbors) > 0:
                # Create pairs (i, j) where j is a neighbor of i
                i_column = torch.full((len(neighbors),), i, device=self.device)
                pairs_i = torch.stack((i_column, neighbors), dim=1)
                pairs.append(pairs_i)
        
        if not pairs:
            return torch.zeros((0, 2), dtype=torch.long, device=self.device)
        
        return torch.cat(pairs, dim=0)
        
    def get_n_neighbors(self):
        """
        Get the number of neighbors for each atom (within the actual cutoff).
        """
        actual_n_neighbors = torch.zeros_like(self.n_neighbors)
        
        for i in range(self.n_atoms):
            n_valid = self.n_neighbors[i]
            if n_valid == 0:
                continue
                
            # Count only neighbors within the actual cutoff
            mask = self.distances[i, :n_valid] < self.cutoff
            actual_n_neighbors[i] = torch.sum(mask)
            
        return actual_n_neighbors