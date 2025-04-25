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
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.npt import NPT
from ase.md.langevin import Langevin
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import fs, bar, GPa, kcal, mol
from ase.stress import full_3x3_to_voigt_6_stress

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
    
    # Loop over the 6 independent components of the stress tensor
    for i in range(3):
        for j in range(3):
            # Create strain tensor (identity + small strain)
            strain = np.eye(3)
            strain[i, j] += epsilon
            
            # Apply strain to the cell
            atoms.set_cell(np.dot(orig_cell, strain), scale_atoms=True)
            
            # Calculate energy after strain
            energy_plus = atoms.get_potential_energy()
            
            # Apply negative strain
            strain = np.eye(3)
            strain[i, j] -= epsilon
            
            # Apply strain to the cell
            atoms.set_cell(np.dot(orig_cell, strain), scale_atoms=True)
            
            # Calculate energy after negative strain
            energy_minus = atoms.get_potential_energy()
            
            # Restore original cell
            atoms.set_cell(orig_cell, scale_atoms=True)
            
            # Central difference formula for the stress component
            virial[i, j] = (energy_plus - energy_minus) / (2.0 * epsilon)
    
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
    virial = torch.matmul(coords.grad.T, coords) + torch.matmul(box.grad.T, box)
    stress = virial / box_volume

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

def test_kinetic_stress():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216_mchem.xyz"), requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )
    ff_ase = CMM_ASE(ff, cm, topology, parameters, output_folder=os.path.join(os.path.dirname(__file__), "scratch"))

    velocities_ref = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/water_216_velocities.txt"))
    kinetic_energy_tensor_ref = np.array(
        [[184.070278950081, 11.441292245682, 3.491837665282],
        [11.441292245682, 202.882792736441, -0.844740236485],
        [3.491837665282, -0.844740236485, 190.565661644341]]
    )
    kinetic_energy_ref = 577.518733330862 # kcal/mol
    assert np.isclose(np.trace(kinetic_energy_tensor_ref), kinetic_energy_ref)
    def get_kinetic_tensor(velocities, masses):
        return 0.5 * np.sum(masses[:, np.newaxis, np.newaxis] * np.array([np.outer(velocities[i], velocities[i]) for i in range(len(velocities))]), axis=0)
    
    conv_fac = 418.4 # I don't know why these are the mchem units?
    kinetic_energy_tensor = get_kinetic_tensor(velocities_ref, ff_ase.atoms.get_masses())
    kinetic_energy_ase_masses = np.trace(kinetic_energy_tensor) / conv_fac # in kcal/mol
    ff_ase.atoms.set_velocities(velocities_ref)
    velocity_conversion = np.sqrt(ff_ase.atoms.get_kinetic_energy() / (kcal / mol) / (kinetic_energy_ase_masses))
    ff_ase.atoms.set_velocities(velocities_ref / velocity_conversion)
    assert np.isclose(ff_ase.atoms.get_kinetic_energy() / (kcal / mol), kinetic_energy_ase_masses)
    kinetic_energy_tensor = get_kinetic_tensor(ff_ase.atoms.get_velocities(), ff_ase.atoms.get_masses())
    kinetic_stress = -2 * kinetic_energy_tensor / (ff_ase.atoms.get_volume())
    kinetic_stress_ase = ff_ase.atoms.get_kinetic_stress()
    assert np.allclose(full_3x3_to_voigt_6_stress(kinetic_stress), kinetic_stress_ase)
    temperature=300
    dyn = NPTBerendsen(ff_ase.atoms, timestep=1.0 * fs, temperature_K=temperature,
               taut=100 * fs, pressure_au=1.01325 * bar,
               taup=1000 * fs, compressibility_au=4.57e-5 / bar)
    def log_step(atoms=ff_ase.atoms):
        energy = atoms.get_potential_energy()
        kinetic = atoms.get_kinetic_energy()
        temperature = atoms.get_temperature()
        print(atoms.get_kinetic_stress())
        stress_total = atoms.get_stress(voigt=True, include_ideal_gas=True)
        stress_virial = atoms.get_stress(voigt=True, include_ideal_gas=False)
        pressure = (-(stress_total[0] + stress_total[1] + stress_total[2]) / 3) / (bar * 1.01325)
        print(stress_total)
        print(stress_virial)
        print(pressure)
        #kinetic_pressure = 2 * kinetic / (3 * atoms.get_volume()) / (bar * 1.01325)
        #print(f"Step: {dyn.nsteps}, E_pot: {energy:.6f} eV, E_kin: {kinetic:.6f} eV, T: {temperature:.1f} K, Virial Press.: {pressure:.2f} atm, Kinetic Press. {kinetic_pressure:.2f}")

    dyn.attach(lambda : log_step(ff_ase.atoms), interval=1)  # Log at every step
    dyn.run(1)

def test_md():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216_mchem.xyz"), requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)
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

    temperature = 300.0
    MaxwellBoltzmannDistribution(ff_ase.atoms, temperature_K=temperature, force_temp=True)
    Stationary(ff_ase.atoms)

    traj = Trajectory('water216_npt_270K.traj', 'a', ff_ase.atoms)

    #dyn = NPT(ff_ase.atoms, timestep=1.0 * fs, temperature_K=temperature, externalstress=1.01325 * bar)
    dyn = NPTBerendsen(ff_ase.atoms, timestep=1.0 * fs, temperature_K=temperature,
                   taut=100 * fs, pressure_au=1.01325 * bar,
                   taup=1000 * fs, compressibility_au=4.57e-5 / bar)
    #dyn = NVTBerendsen(ff_ase.atoms, timestep=1.0 * fs, temperature_K=temperature, taut=0.5*1000*fs)
    #dyn = VelocityVerlet(ff_ase.atoms, 1.0 * fs)
    #dyn = Langevin(ff_ase.atoms, timestep=1.0 * fs, temperature_K=temperature, friction=0.01 / fs)
    dyn.attach(traj.write, interval=1)
    logger.attach_to_ase_dynamics(dyn)
    energy_logger.attach_to_ase_dynamics(dyn)

    def log_step(atoms=ff_ase.atoms):
        energy = atoms.get_potential_energy()
        kinetic = atoms.get_kinetic_energy()
        temperature = atoms.get_temperature()
        print(atoms.get_kinetic_stress())
        stress = atoms.get_stress(voigt=True, include_ideal_gas=True)
        pressure = (-(stress[0] + stress[1] + stress[2]) / 3) / (bar * 1.01325)
        kinetic_pressure = 2 * kinetic / (3 * atoms.get_volume()) / (bar * 1.01325)
        print(f"Step: {dyn.nsteps}, E_pot: {energy:.6f} eV, E_kin: {kinetic:.6f} eV, T: {temperature:.1f} K, Virial Press.: {pressure:.2f} atm, Kinetic Press. {kinetic_pressure:.2f}")

    dyn.attach(lambda : log_step(ff_ase.atoms), interval=1)  # Log at every step
    dyn.run(100)

def test_long_range_dispersion_correction():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216_mchem.xyz"), requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True,use_lr_dispersion_correction=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )
    energies = ff.evaluate(cm, topology, parameters)
    print(energies)
