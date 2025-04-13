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
from cmm.logger import Logger
from cmm.memory import MemoryTracker, add_memory_tracking_to_md, BackwardMemoryMonitor, patch_cmm_ase_with_backward_memory_monitor

from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.md import VelocityVerlet, MDLogger
from ase.md.nptberendsen import NPTBerendsen
from ase.md.langevin import Langevin
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import fs, bar, GPa

def calculate_virial_finite_difference(atoms, epsilon=0.001):
    """
    Calculate the virial tensor using finite difference by applying small strains
    to the unit cell and measuring the energy change.
    
    Args:
        atoms: ASE atoms object with a calculator
        epsilon: Finite difference strain step size
    
    Returns:
        stress: 3x3 stress tensor in eV/Å³
    """
    virial = np.zeros((3, 3))
    orig_cell = atoms.get_cell().copy()
    #orig_volume = atoms.get_volume()
    
    # Loop over the 6 independent components of the stress tensor
    for i in range(3):
        for j in range(3):
            # Create strain tensor (identity + small strain)
            strain = np.eye(3)
            strain[i, j] += epsilon
            #strain[j, i] += epsilon
            
            # Apply strain to the cell
            atoms.set_cell(np.dot(orig_cell, strain), scale_atoms=True)
            
            # Calculate energy after strain
            energy_plus = atoms.get_potential_energy()
            
            # Apply negative strain
            strain = np.eye(3)
            strain[i, j] -= epsilon
            #strain[j, i] -= epsilon
            
            # Apply strain to the cell
            atoms.set_cell(np.dot(orig_cell, strain), scale_atoms=True)
            
            # Calculate energy after negative strain
            energy_minus = atoms.get_potential_energy()
            
            # Restore original cell
            atoms.set_cell(orig_cell, scale_atoms=True)
            
            # Central difference formula for the stress component
            # Extra factor of 2 since applying strain symmetrically
            virial[i, j] = (energy_plus - energy_minus) / (2.0 * epsilon)
            #virial[j, i] = virial[i, j]  # Stress tensor is symmetric
    
    return virial

def test_virial_tensor():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)
    box_volume = torch.det(box)

    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies = ff.evaluate(cm, topology, parameters)
    energies['total'].backward()

    # This equation for the virial stress is derived in the Appendix of: https://doi.org/10.1016/j.cpc.2019.107057
    # This is equivalent to taking the outer product of each particle gradient with the particle position
    # plus each box gradient with the box vector. (i.e. sum over F cross R) Note that the box is just
    # another degree of freedom in the simulation.
    print(torch.matmul(coords.grad.T, coords))
    print(torch.matmul(box.grad.T, box))
    virial = torch.matmul(coords.grad.T, coords) + torch.matmul(box.grad.T, box)
    stress = virial / box_volume
    print(stress)

    coords, atom_types, bonds, _ = read_from_tinker_xyz(system_file, requires_grad=False, device=device)
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=False, device=device)
    box_volume = torch.det(box)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    ff_ase = CMM_ASE(ff, cm, topology, parameters)

    virial_fd = torch.from_numpy(calculate_virial_finite_difference(ff_ase.atoms, epsilon=1e-6) / HARTREE2EV).to(device)
    stress_fd = virial_fd / box_volume

    assert torch.allclose(stress_fd, stress)

def test_virial_tensor_translational_invariance():
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
    energies['total'].backward()

    virial = torch.matmul(coords.grad.T, coords) + torch.matmul(box.grad.T, box)
    stress = virial / box_volume

    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True)
    
    # Translation along all x, y, and z
    coords = coords + torch.ones_like(coords) * torch.rand(3, dtype=torch.float64) * 10.0
    coords = coords.detach().clone().requires_grad_()
    
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

    energies_2 = ff.evaluate(cm, topology, parameters)
    energies_2['total'].backward()

    virial_2 = torch.matmul(coords.grad.T, coords) + torch.matmul(box.grad.T, box)
    stress_2 = virial_2 / box_volume

    assert torch.allclose(stress, stress_2)
    assert torch.isclose(energies['total'], energies_2['total'])

def test_md():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=False, device=device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )
    ff_ase = CMM_ASE(ff, cm, topology, parameters, output_folder=os.path.join(os.path.dirname(__file__), "scratch"))

    logger = Logger(
        cm=cm,
        ff=ff,
        ase_calculator=ff_ase,
        log_interval=1,
        output_folder=ff_ase.output_folder,
        log_file="water216_npt_270K.log",
        properties=[
            "step", "temperature", "energy_total",
            "kinetic_energy", "volume", "density", "pressure",
            "dipole_magnitude"
        ]
    )

    energy_logger = Logger(
        cm=cm,
        ff=ff,
        ase_calculator=ff_ase,
        log_interval=1,
        output_folder=ff_ase.output_folder,
        log_file="water216_npt_270K_energies.log",
        properties=[
            "step", "kinetic_energy",
            "energy_total", "energy_perm_elec", "energy_pol", "energy_ct_direct",
            "energy_xpol", "energy_pauli", "energy_disp", "energy_deformation",
            "energy_bond", "energy_angle", "energy_bond_bond", "energy_bond_angle",
            "energy_ewald",
        ]
    )

    temperature = 270.0
    MaxwellBoltzmannDistribution(ff_ase.atoms, temperature_K=temperature, force_temp=True)
    Stationary(ff_ase.atoms)

    traj = Trajectory('water216_npt_270K.traj', 'a', ff_ase.atoms)

    dyn = NPTBerendsen(ff_ase.atoms, timestep=1.0 * fs, temperature_K=temperature,
                   taut=100 * fs, pressure_au=1.01325 * bar,
                   taup=1000 * fs, compressibility_au=4.57e-5 / bar)
    dyn.attach(traj.write, interval=1)
    logger.attach_to_ase_dynamics(dyn)
    energy_logger.attach_to_ase_dynamics(dyn)

    def log_step(atoms=ff_ase.atoms):
        energy = atoms.get_potential_energy()
        kinetic = atoms.get_kinetic_energy()
        temperature = atoms.get_temperature()
        stress = atoms.get_stress(voigt=True)
        print(stress)
        print(bar * 1.01325)
        pressure = (-(stress[0] + stress[1] + stress[2]) / 3) / (bar * 1.01325)
        print(f"Step: {dyn.nsteps}, E_pot: {energy:.6f} eV, E_kin: {kinetic:.6f} eV, T: {temperature:.1f} K, P: {pressure:.2f} atm")

    dyn.attach(lambda : log_step(ff_ase.atoms), interval=1)  # Log at every step
    dyn.run(100)

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

def test_memory_usage():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=False, device=device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )
    ff_ase = CMM_ASE(ff, cm, topology, parameters, output_folder=os.path.join(os.path.dirname(__file__), "scratch"))
    #mt = MemoryTracker(os.path.join(os.path.dirname(__file__), "scratch/memory_logs"))
    #ff_ase = add_memory_tracking_to_md(ff_ase, mt, check_interval=5)
    backward_monitor = patch_cmm_ase_with_backward_memory_monitor(ff_ase)

    temperature = 270.0
    MaxwellBoltzmannDistribution(ff_ase.atoms, temperature_K=temperature, force_temp=True)
    Stationary(ff_ase.atoms)

    dyn = NPTBerendsen(ff_ase.atoms, timestep=1.0 * fs, temperature_K=temperature,
                   taut=100 * fs, pressure_au=1.01325 * bar,
                   taup=1000 * fs, compressibility_au=4.57e-5 / bar)

    def log_step(atoms=ff_ase.atoms):
        energy = atoms.get_potential_energy()
        kinetic = atoms.get_kinetic_energy()
        temperature = atoms.get_temperature()
        stress = atoms.get_stress(voigt=True)
        print(stress)
        print(bar * 1.01325)
        pressure = (-(stress[0] + stress[1] + stress[2]) / 3) / (bar * 1.01325)
        print(f"Step: {dyn.nsteps}, E_pot: {energy:.6f} eV, E_kin: {kinetic:.6f} eV, T: {temperature:.1f} K, P: {pressure:.2f} atm")

    #dyn.attach(lambda : log_step(ff_ase.atoms), interval=1)  # Log at every step
    dyn.run(100)
    #mt.finalize(limit_tensor_report=True)
    backward_monitor.summarize()