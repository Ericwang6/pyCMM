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

from ase.optimize import LBFGS
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import fs

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
    virial = torch.matmul(box.grad.T, box) + torch.matmul(coords.grad.T, coords)
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

    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), requires_grad=True)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True)
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
    print(energies['tot'])
    print(coords.grad)

def test_npt_optimization():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), device=device)
    permutation = np.argsort(bonds[0], kind='stable') # Make sure sort is stable so equivalent indices don't get swapped.
    bonds[0] = bonds[0][permutation]
    bonds[1] = bonds[1][permutation]
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 100.0 / BOHR2ANG, requires_grad=False, device=device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=False, cutoff_short_range=9.0 / BOHR2ANG)
    with torch.no_grad():
        topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )

    calculator = CMM_ASE(ff, cm, topology, parameters, output_folder=os.path.join(os.path.dirname(__file__), "scratch"))

    # Initial forces calculation
    initial_forces = calculator.atoms.get_forces()
    print(f"Initial max force: {np.max(np.abs(initial_forces))}")

    temperature = 150.0  # K
    MaxwellBoltzmannDistribution(calculator.atoms, temperature_K=temperature, force_temp=True)
    #Stationary(calculator.atoms)
    #ZeroRotation(calculator.atoms)

    def log_step(atoms=calculator.atoms):
        energy = atoms.get_potential_energy()
        kinetic = atoms.get_kinetic_energy()
        temperature = atoms.get_temperature()
        print(f"Step: {dyn.nsteps}, E_pot: {energy:.6f} eV, E_kin: {kinetic:.6f} eV, T: {temperature:.1f} K")

    dyn = VelocityVerlet(calculator.atoms, 0.5 * fs, trajectory=os.path.join(os.path.dirname(__file__), 'scratch/w2_dynamics.traj'))
    dyn.attach(lambda : log_step(calculator.atoms), interval=10)  # Log at every step
    dyn.attach(calculator.create_checkpoint, interval=50)  # Checkpoint every 50 steps
    finished = dyn.run(100)
    if finished:
        calculator.save_state(os.path.join(os.path.dirname(__file__), 'scratch/final_state.json'))

    # To restart from a checkpoint
    calculator = CMM_ASE.load_state(os.path.join(os.path.dirname(__file__), 'scratch/final_state.json'))
    dyn = VelocityVerlet(calculator.atoms, 0.5 * fs, trajectory=os.path.join(os.path.dirname(__file__), 'scratch/w2_dynamics.traj'))
    dyn.attach(lambda : log_step(calculator.atoms), interval=10)  # Log at every step
    dyn.attach(calculator.create_checkpoint, interval=50)  # Checkpoint every 5 steps
    finished = dyn.run(100)
    if finished:
        calculator.save_state(os.path.join(os.path.dirname(__file__), 'scratch/final_state_2.json'))