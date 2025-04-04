import pytest
import torch
import numpy as np
import os

from cmm.units import HARTREE2EV, BOHR2ANG, BOHR2NM
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM
from cmm.interfaces import CMM_ASE

from ase.optimize import BFGS

def calculate_stress_finite_difference(atoms, epsilon=0.001):
    """
    Calculate the stress tensor using finite difference by applying small strains
    to the unit cell and measuring the energy change.
    
    Args:
        atoms: ASE atoms object with a calculator
        epsilon: Finite difference strain step size
    
    Returns:
        stress: 3x3 stress tensor in eV/Å³
    """
    stress = np.zeros((3, 3))
    orig_cell = atoms.get_cell().copy()
    orig_volume = atoms.get_volume()
    
    # Loop over the 6 independent components of the stress tensor
    for i in range(3):
        for j in range(i, 3):
            # Create strain tensor (identity + small strain)
            strain = np.eye(3)
            strain[i, j] += epsilon
            strain[j, i] += epsilon
            
            # Apply strain to the cell
            atoms.set_cell(np.dot(orig_cell, strain), scale_atoms=True)
            
            # Calculate energy after strain
            energy_plus = atoms.get_potential_energy()
            
            # Apply negative strain
            strain = np.eye(3)
            strain[i, j] -= epsilon
            strain[j, i] -= epsilon
            
            # Apply strain to the cell
            atoms.set_cell(np.dot(orig_cell, strain), scale_atoms=True)
            
            # Calculate energy after negative strain
            energy_minus = atoms.get_potential_energy()
            
            # Restore original cell
            atoms.set_cell(orig_cell, scale_atoms=True)
            
            # Central difference formula for the stress component
            stress[i, j] = (energy_plus - energy_minus) / (2.0 * epsilon) / (2.0 * orig_volume)
            stress[j, i] = stress[i, j]  # Stress tensor is symmetric
    
    return stress

def test_virial_tensor():
    torch.set_default_dtype(torch.float64)

    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True)
    box_volume = torch.det(box)

    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM()
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies = ff.evaluate(cm, topology, parameters)
    energies['tot'].backward()

    # This equation for the virial stress is derived in the Appendix of: https://doi.org/10.1016/j.cpc.2019.107057
    # This is equivalent to taking the outer product of each particle gradient with the particle position
    # plus each box gradient with the box vector. (i.e. sum over F cross R) Note that the box is just
    # another degree of freedom in the simulation.
    right = torch.matmul(box.grad.T, box)
    left = torch.matmul(coords.grad.T, coords)
    virial = right + left
    stress = virial / box_volume

    coords, atom_types, bonds, _ = read_from_tinker_xyz(system_file, requires_grad=False)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM()
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    ff_ase = CMM_ASE(ff, cm, topology, parameters)

    stress_fd = torch.from_numpy(calculate_stress_finite_difference(ff_ase.atoms, epsilon=1e-6) * BOHR2ANG**3 / HARTREE2EV)
    assert torch.allclose(stress_fd, stress)

def test_md():
    torch.set_default_dtype(torch.float64)

    coords, atom_types, bonds, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), requires_grad=True)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM()
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies = ff.evaluate(cm, topology, parameters)
    energies['tot'].backward()
    print(energies['tot'])
    print(coords.grad)

def test_optimize_nacl():
    torch.set_default_dtype(torch.float64)
    torch.autograd.set_detect_anomaly(True)

    positions = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/nacl_crystal.txt"), dtype=np.float64)
    positions[0, 0] += 0.1 # move from equilibrium
    coords = torch.tensor(positions / BOHR2NM, requires_grad=True)
    bonds = np.array([], dtype=np.float64)
    atom_type_names = ["" for i in range(coords.size(0))]
    for i in range(coords.size(0) // 2):
        atom_type_names[i] = "Na+"
    for i in range(coords.size(0) // 2, coords.size(0)):
        atom_type_names[i] = "Cl-"

    box = torch.tensor(np.eye(3) * 28.2 / BOHR2ANG, requires_grad=True)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 2048)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    with torch.no_grad():
        ff = CMM(cutoff_ewald=torch.tensor(10.0 / BOHR2ANG), ewald_tolerance=torch.tensor(1e-5), use_ewald=True)
        topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )

    ff.alpha_damp_exponent[3] = 0.0 # 3 corresponds to Cl-
    ff.alpha_damp_max[3] = 0.0
    ff.rebuild_atomic_params()

    ff_ase = CMM_ASE(ff, cm, topology, parameters)
    ff_ase.calculate()
    dyn = BFGS(ff_ase.atoms, trajectory='nacl_opt.traj')
    dyn.run(fmax=1e-6, steps=10)