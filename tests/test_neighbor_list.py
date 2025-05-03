import torch
import pytest
import numpy as np
from cmm.neighbor_list import NSquaredList, NeighborList, VerletList
from cmm.units import HARTREE2EV, BOHR2ANG, BOHR2NM
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM
from cmm.interfaces import CMM_ASE
from ase.units import Bohr
import os
import json

def check_neighbor_lists_match(ref_neighbors: NeighborList, true_neighbors: NeighborList, num_atoms: int):
    """
    Check if cell list neighbors match true neighbors within max_neighbors limit.
    
    Args:
        ref_neighbors (NeighborList): neighbor list derived from CellList most likely
        true_neighbors (NeighborList): Probably an NSquaredList
        
    Returns:
        bool: True if all neighbor lists match within limits
    """
    equal_nls = False
    for i in range(num_atoms):
        ref_neighs = ref_neighbors.get_neighbors(i)
        true_neighs = true_neighbors.get_neighbors(i)

        try:
            equal_nls |= (ref_neighs.size(0) == true_neighs.size(0))
            equal_nls |= torch.all(ref_neighs == true_neighs)
        except:
            equal_nls = False
            break
    return equal_nls

def test_neighbor_list_rebuild_water():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    assert check_neighbor_lists_match(ff_ase_1._cm.neighbor_list, ff_ase_2._cm.neighbor_list, len(positions_np) // 3)
    assert torch.isclose(ff_ase_1._energies['total'], ff_ase_2._energies['total'])
    assert np.allclose(ff_ase_1.get_forces(), ff_ase_2.get_forces(), rtol=1e-6, atol=1e-6)

def test_nsquared_neighbor_list():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nl_1_file = os.path.join(os.path.dirname(__file__), 'data/config_1_nl.json')
    nl_2_file = os.path.join(os.path.dirname(__file__), 'data/config_2_nl.json')
    with open(nl_1_file, 'r') as f:
        state_1 = json.load(f)
    with open(nl_2_file, 'r') as f:
        state_2 = json.load(f)
    
    # Load reference data #
    # state 1
    coords_1 = torch.tensor(state_1["positions"], device=device)
    cell_1 = torch.tensor(state_1["cell"], device=device)
    pairs_1 = torch.tensor(state_1["pairs"], device=device, dtype=torch.long)
    pairs_1 = pairs_1 - torch.ones_like(pairs_1)
    # ^^^ Subtract one since the reference code uses one-based indexing
    # state 2
    coords_2 = torch.tensor(state_2["positions"], device=device)
    cell_2 = torch.tensor(state_2["cell"], device=device)
    pairs_2 = torch.tensor(state_2["pairs"], device=device, dtype=torch.long)
    pairs_2 = pairs_2 - torch.ones_like(pairs_2)
    # ^^^ Subtract one since the reference code uses one-based indexing
    cutoff = torch.tensor(4.0, device=device)

    # Build two separate neighbor lists for each state #
    nl_1 = NSquaredList(coords_1, cell_1, cutoff)
    nl_2 = NSquaredList(coords_2, cell_2, cutoff)
    pairs_nl_1 = nl_1.get_pairs()
    pairs_nl_2 = nl_2.get_pairs()
    _, sorted_indices_nl_1 = torch.sort(pairs_nl_1[:,0])
    _, sorted_indices_nl_2 = torch.sort(pairs_nl_2[:,0])
    pairs_sorted_nl_1 = torch.stack((pairs_nl_1[:,0][sorted_indices_nl_1], pairs_nl_1[:,1][sorted_indices_nl_1]))
    pairs_sorted_nl_2 = torch.stack((pairs_nl_2[:,0][sorted_indices_nl_2], pairs_nl_2[:,1][sorted_indices_nl_2]))
    
    _, sorted_indices_1 = torch.sort(pairs_1[:,0])
    _, sorted_indices_2 = torch.sort(pairs_2[:,0])
    pairs_sorted_1 = torch.stack((pairs_1[:,0][sorted_indices_1], pairs_1[:,1][sorted_indices_1]))
    pairs_sorted_2 = torch.stack((pairs_2[:,0][sorted_indices_2], pairs_2[:,1][sorted_indices_2]))
    assert torch.equal(pairs_sorted_1[0,:], pairs_sorted_nl_1[0,:])
    assert torch.sum(pairs_sorted_1[1,:] - pairs_sorted_nl_1[1,:]) == 0
    assert torch.equal(pairs_sorted_2[0,:], pairs_sorted_nl_2[0,:])
    assert torch.sum(pairs_sorted_2[1,:] - pairs_sorted_nl_2[1,:]) == 0

    # Now test if we can update the neighbor list succesffully from state 1 to state 2 #
    assert check_neighbor_lists_match(nl_1, nl_2, nl_1.natoms) == False
    nl_1.update(coords_2, cell_2)
    assert check_neighbor_lists_match(nl_1, nl_2, nl_1.natoms) == True

    # Now check that we can successfully rebuild when the cell changes size
    nl_1 = NSquaredList(coords_1, cell_1, cutoff)
    new_cell = cell_1 + torch.eye(3)
    nl_2 = NSquaredList(coords_1, new_cell, cutoff)
    assert check_neighbor_lists_match(nl_1, nl_2, nl_1.natoms) == False
    nl_1.update(coords_1, new_cell)
    assert check_neighbor_lists_match(nl_1, nl_2, nl_1.natoms) == True

    # Now check that we can successfully rebuild when the cutoff changes
    nl_1 = NSquaredList(coords_1, cell_1, cutoff)
    nl_2 = NSquaredList(coords_1, cell_1, cutoff + 2.0)
    assert check_neighbor_lists_match(nl_1, nl_2, nl_1.natoms) == False
    nl_1.update(coords_1, cell_1, cutoff + 2.0)
    assert check_neighbor_lists_match(nl_1, nl_2, nl_1.natoms) == True

def test_verlet_neighbor_list():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nl_1_file = os.path.join(os.path.dirname(__file__), 'data/config_1_nl.json')
    nl_2_file = os.path.join(os.path.dirname(__file__), 'data/config_2_nl.json')
    with open(nl_1_file, 'r') as f:
        state_1 = json.load(f)
    with open(nl_2_file, 'r') as f:
        state_2 = json.load(f)
    
    # Load reference data #
    # state 1
    coords_1 = torch.tensor(state_1["positions"], device=device)
    cell_1 = torch.tensor(state_1["cell"], device=device)
    pairs_1 = torch.tensor(state_1["pairs"], device=device, dtype=torch.long)
    pairs_1 = pairs_1 - torch.ones_like(pairs_1)
    # ^^^ Subtract one since the reference code uses one-based indexing
    # state 2
    coords_2 = torch.tensor(state_2["positions"], device=device)
    cell_2 = torch.tensor(state_2["cell"], device=device)
    pairs_2 = torch.tensor(state_2["pairs"], device=device, dtype=torch.long)
    pairs_2 = pairs_2 - torch.ones_like(pairs_2)
    # ^^^ Subtract one since the reference code uses one-based indexing
    cutoff = torch.tensor(4.0, device=device)

    # Build two separate neighbor lists for each state #
    nl_1_ref = NSquaredList(coords_1, cell_1, cutoff)
    nl_2_ref = NSquaredList(coords_2, cell_2, cutoff)
    pairs_nl_1_ref = nl_1_ref.get_pairs()
    pairs_nl_2_ref = nl_2_ref.get_pairs()

    nl_1_verlet = VerletList(coords_1, cell_1, cutoff, torch.tensor(1.0))
    pairs_nl_1_verlet = nl_1_verlet.get_pairs()
    assert check_neighbor_lists_match(nl_1_verlet, nl_1_ref, nl_1_verlet.natoms)
    assert torch.equal(nl_1_ref.get_pairs(), nl_1_verlet.get_pairs())

    # Test full update
    assert check_neighbor_lists_match(nl_1_verlet, nl_2_ref, nl_1_verlet.natoms) == False
    nl_1_verlet.update(coords_2, cell_2)
    assert check_neighbor_lists_match(nl_1_verlet, nl_2_ref, nl_1_verlet.natoms) == True    

    # Test update from existing pair list inside verlet cutoff
    nl_1_verlet = VerletList(coords_1, cell_1, cutoff, torch.tensor(1.0))
    coords_inside_verlet_cutoff = torch.zeros_like(coords_1)
    factor = 1.0
    while True:
        coords_inside_verlet_cutoff = coords_1 + torch.rand_like(coords_1) * factor
        max_displacement = torch.max(torch.abs(coords_inside_verlet_cutoff - coords_1))
        if max_displacement < 0.5 * nl_1_verlet._padding:
            break
        factor *= 0.5
    
    nl_1_ref = NSquaredList(coords_inside_verlet_cutoff, cell_1, cutoff)
    nl_1_verlet.update(coords_inside_verlet_cutoff, cell_1)
    assert check_neighbor_lists_match(nl_1_verlet, nl_1_ref, nl_1_verlet.natoms) == True    
    assert torch.equal(nl_1_ref.get_pairs(), nl_1_verlet.get_pairs())

    nl_verlet = VerletList(coords_2, cell_2, cutoff, torch.tensor(1.0))
    modified_cell = cell_2 - torch.eye(3) * 0.49
    nl_ref = NSquaredList(coords_2, modified_cell, cutoff)
    nl_verlet.update(coords_2, modified_cell)
    assert check_neighbor_lists_match(nl_verlet, nl_ref, nl_verlet.natoms) == True    
    assert torch.equal(nl_ref.get_pairs(), nl_verlet.get_pairs())
    assert torch.equal(nl_verlet._reference_box, cell_2)

    # Check we properly update the reference box on rebuild
    modified_cell = cell_2 - torch.eye(3) * 0.51
    nl_ref = NSquaredList(coords_2, modified_cell, cutoff)
    nl_verlet.update(coords_2, modified_cell)
    assert torch.equal(nl_ref.get_pairs(), nl_verlet.get_pairs())
    assert torch.equal(nl_verlet._reference_box, modified_cell)

    # Test we properly update when both positions and the cell change
    nl_verlet = VerletList(coords_1, cell_1, cutoff, torch.tensor(1.0))
    coords_inside_verlet_cutoff = torch.zeros_like(coords_1)
    factor = 1.0
    while True:
        coords_inside_verlet_cutoff = coords_1 + torch.rand_like(coords_1) * factor
        max_displacement = torch.max(torch.abs(coords_inside_verlet_cutoff - coords_1))
        if max_displacement < 0.25 * nl_verlet._padding:
            break
        factor *= 0.5

    # Update from verlet list
    modified_cell = cell_1 - torch.eye(3) * 0.24
    nl_ref = NSquaredList(coords_inside_verlet_cutoff, modified_cell, cutoff)
    nl_verlet.update(coords_inside_verlet_cutoff, modified_cell)
    assert check_neighbor_lists_match(nl_verlet, nl_ref, nl_verlet.natoms) == True    
    assert torch.equal(nl_ref.get_pairs(), nl_verlet.get_pairs())
    assert torch.equal(nl_verlet._reference_positions, coords_1)
    assert torch.equal(nl_verlet._reference_box, cell_1)

    # Update by full rebuild
    modified_cell = cell_1 - torch.eye(3) * 0.26
    nl_ref = NSquaredList(coords_inside_verlet_cutoff, modified_cell, cutoff)
    nl_verlet.update(coords_inside_verlet_cutoff, modified_cell)
    assert check_neighbor_lists_match(nl_verlet, nl_ref, nl_verlet.natoms) == True    
    assert torch.equal(nl_ref.get_pairs(), nl_verlet.get_pairs())
    assert torch.equal(nl_verlet._reference_positions, coords_inside_verlet_cutoff)
    assert torch.equal(nl_verlet._reference_box, modified_cell)