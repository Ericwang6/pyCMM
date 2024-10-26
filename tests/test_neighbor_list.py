import torch
import pytest
import numpy as np
from cmm.neighbor_list import CellList, NSquaredList, NeighborList

def generate_random_system(n_particles, density, seed=None):
    """
    Generate random particle positions in a cubic box.
    
    Args:
        n_particles (int): Number of particles
        box_size (float): Size of cubic box
        density (float, optional): If provided, override box_size to match density
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        tuple: (positions, box_vectors)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    volume = n_particles / density
    box_size = volume ** (1/3)
    
    positions = torch.rand(n_particles, 3) * box_size
    box_vectors = torch.tensor([box_size, box_size, box_size])
    
    return positions, box_vectors


def check_neighbor_lists_match(ref_neighbors: NeighborList, true_neighbors: NeighborList, num_atoms: int):
    """
    Check if cell list neighbors match true neighbors within max_neighbors limit.
    
    Args:
        ref_neighbors (NeighborList): neighbor list derived from CellList most likely
        true_neighbors (NeighborList): Probably an NSquaredList
        
    Returns:
        bool: True if all neighbor lists match within limits
    """
    for i in range(num_atoms):
        ref_neighs = ref_neighbors.get_neighbors(i)
        true_neighs = true_neighbors.get_neighbors(i)

        assert torch.all(ref_neighs == true_neighs), f"Neighbor mismatch for particle {i}"

@pytest.mark.parametrize("n_particles", [100, 1000])
@pytest.mark.parametrize("density", [0.5, 2.0])
@pytest.mark.parametrize("cutoff", [0.5, 2.0])
def test_cell_list_random_system(n_particles, density, cutoff):
    """Test cell list with random particle positions at different densities."""
    # Generate random system
    positions, box_vectors = generate_random_system(n_particles, density=density, seed=42)
    
    # Create cell list and reference list
    cell_list = CellList(positions, box_vectors, cutoff, max_neighbors=2048)
    nsq_list = NSquaredList(positions, box_vectors, cutoff)
    
    check_neighbor_lists_match(cell_list, nsq_list, nsq_list.natoms)

def test_cell_list_update():
    # Generate initial system
    n_particles = 100
    cutoff = 2.0
    max_neighbors = 2048
    
    positions, box_vectors = generate_random_system(n_particles, density=1.0, seed=42)
    
    # Create cell list
    cell_list = CellList(positions, box_vectors, cutoff, max_neighbors=max_neighbors)
    
    # Make small random displacements
    displacements = (torch.rand_like(positions) - 0.5 * cutoff) * 0.1
    new_positions = positions + displacements
    
    nsq_list = NSquaredList(new_positions, box_vectors, cutoff)

    # Update cell list
    cell_list.update(new_positions)
    
    # Check new neighbor lists
    check_neighbor_lists_match(cell_list, nsq_list, nsq_list.natoms)

if __name__ == "__main__":
    pytest.main([__file__])