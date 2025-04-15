import torch
import pytest
import numpy as np
from cmm.neighbor_list import CellList, NSquaredList, NeighborList
from cmm.units import HARTREE2EV, BOHR2ANG, BOHR2NM
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM
from cmm.interfaces import CMM_ASE
from ase.units import Bohr

import os

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
    displacements = (torch.rand_like(positions) - cutoff)
    new_positions = positions + displacements
    
    nsq_list = NSquaredList(new_positions, box_vectors, cutoff)

    # Update cell list
    cell_list.update(new_positions, box_vectors)
    
    # Check new neighbor lists
    check_neighbor_lists_match(cell_list, nsq_list, nsq_list.natoms)

def test_neighbor_list_rebuild_water():
    torch.set_default_dtype(torch.float64)
    ff_ase_1 = CMM_ASE.load_state(os.path.join(os.path.dirname(__file__), 'data/checkpoint_20250413_023453.json'))
    ff_ase_2 = CMM_ASE.load_state(os.path.join(os.path.dirname(__file__), 'data/checkpoint_20250413_031637.json'))
    #print("----- Calculation on state 1 -----")
    ff_ase_1.calculate()
    #print("----- Calculation on state 2 -----")
    ff_ase_2.calculate()

    positions_np = ff_ase_2.atoms.get_positions()
    box_np = ff_ase_2.atoms.get_cell().array
    ff_ase_1.atoms.set_positions(positions_np)
    ff_ase_1.atoms.set_cell(box_np)
    #print("----- Calculation on state 2 with calculator 1 -----")
    ff_ase_1.calculate()
    assert ff_ase_1._cm.cutoff == ff_ase_2._cm.cutoff
    assert ff_ase_1._cm.neighbor_list.cutoff == ff_ase_2._cm.neighbor_list.cutoff
    assert np.allclose(ff_ase_1.atoms.get_positions(), ff_ase_2.atoms.get_positions())
    assert np.allclose(ff_ase_1.atoms.get_cell().array, ff_ase_2.atoms.get_cell().array)
    assert torch.allclose(ff_ase_1._cm.coords, ff_ase_2._cm.coords)
    assert torch.allclose(ff_ase_1._cm.box, ff_ase_2._cm.box)
    check_neighbor_lists_match(ff_ase_1._cm.neighbor_list, ff_ase_2._cm.neighbor_list, len(positions_np) // 3)
    assert torch.isclose(ff_ase_1._energies['total'], ff_ase_2._energies['total'])
    assert np.allclose(ff_ase_1.get_forces(), ff_ase_2.get_forces(), rtol=1e-6, atol=1e-6)

def test_verlet_list():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True, device=device)
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)
    cutoff = 4.0 / BOHR2ANG
    cm = CoordinateManager(coords, box, cutoff, labels=labels, max_neighbors=1024)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()

    nsq_list = NSquaredList(coords, torch.tensor([18.643, 18.643, 18.643]) / BOHR2ANG, cutoff)
    check_neighbor_lists_match(cm.neighbor_list, nsq_list, nsq_list.natoms)
    assert torch.allclose(nsq_list.get_pairs(), cm.neighbor_list.get_pairs())

    # Test that we can update coordinates in the CM and rebuild properly #
    displacements = torch.rand_like(coords) * 2
    new_coords = coords + displacements
    nsq_list = NSquaredList(new_coords, torch.tensor([18.643, 18.643, 18.643])/ BOHR2ANG, cutoff)
    cm.update_coordinates(new_coords)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    # ^^^ Note that calling this triggers a neighbor list rebuild if needed
    check_neighbor_lists_match(cm.neighbor_list, nsq_list, nsq_list.natoms)
    assert torch.allclose(nsq_list.get_pairs(), cm.neighbor_list.get_pairs())
    
    # Test that we can update coordinates and box in the CM and rebuild properly #
    displacements = torch.rand_like(coords) * 2
    new_coords = coords + displacements
    new_box = torch.tensor(np.eye(3) * 18.0 / BOHR2ANG, requires_grad=True, device=device)
    nsq_list = NSquaredList(new_coords, torch.tensor([18.0, 18.0, 18.0])/ BOHR2ANG, cutoff)
    cm.update_coordinates(new_coords)
    cm.update_box(new_box)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    # ^^^ Note that calling this triggers a neighbor list rebuild if needed
    check_neighbor_lists_match(cm.neighbor_list, nsq_list, nsq_list.natoms)
    assert torch.allclose(nsq_list.get_pairs(), cm.neighbor_list.get_pairs())


if __name__ == "__main__":
    pytest.main([__file__])