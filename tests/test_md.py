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

def test_pbc():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/na_cl.xyz"), requires_grad=True)
    atom_indices_to_names = {
        0: "O_water", 1: "H_water",
        2: "F-", 3: "Cl-", 4: "Br-", 5: "I-",
        6: "Li+", 7: "Na+", 8: "K+", 9: "Rb+", 10: "Cs+",
        11: "Mg2+", 12: "Ca2+"
    }
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 20, requires_grad=False)
    
    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0, labels, topology.all_intramolecular_pairs)
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM()
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )
    ff_ase = CMM_ASE(ff, cm, topology, parameters)
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()
    ff_ase.atoms.set_positions(ff_ase.atoms.get_positions() + np.array([[0.0, 0.0, 0.0],[BOHR2ANG, 0.0, 0.0]]))
    ff_ase.calculate()

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

    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True, use_lr_dispersion=True)
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
    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True, use_lr_dispersion=True)
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True)
    box_volume = torch.det(box)

    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
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

    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
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
    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
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

#def test_polarization_solve_strategies():
#    torch.set_default_dtype(torch.float64)
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216_mchem.xyz"), requires_grad=True, device=device)
#    
#    # Normally, the parser should enforce just returning the names of atom types
#    atom_indices_to_names = {0: "O_water", 1: "H_water"}
#    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
#    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)
#    topology = Topology(bonds, coords.size(0), device)
#    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
#    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
#    ff = CMM(use_ewald=True, use_lr_dispersion=True, solve_tolerance=1e-5)
#    parameters = Parameterizer(
#        atom_type_names, pairs, topology.angle_atoms,
#        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
#    )
#    ff_ase = CMM_ASE(ff, cm, topology, parameters, output_folder=os.path.join(os.path.dirname(__file__), "scratch"))
#
#    temperature = 300.0
#    MaxwellBoltzmannDistribution(ff_ase.atoms, temperature_K=temperature, force_temp=True)
#    Stationary(ff_ase.atoms)
#
#    dyn = VelocityVerlet(ff_ase.atoms, 1.0 * fs)
#    #dyn = Langevin(ff_ase.atoms, timestep=1.0 * fs, temperature_K=temperature, friction=0.01 / fs)
#
#    def log_step(ff_ase):
#        print(ff_ase._ff.polarization_solver.n_solves)
#
#    #dyn.attach(lambda : log_step(ff_ase), interval=1)  # Log at every step
#    dyn.run(60)

#def test_induced_multipole_derivatives():
#    torch.set_default_dtype(torch.float64)
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=True, device=device)
#    
#    # Normally, the parser should enforce just returning the names of atom types
#    atom_indices_to_names = {0: "O_water", 1: "H_water"}
#    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
#    box = torch.tensor(np.eye(3) * 100.0 / BOHR2ANG, requires_grad=True, device=device)
#    topology = Topology(bonds, coords.size(0), device)
#    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
#    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
#    ff = CMM(solve_tolerance=torch.tensor(1e-12))
#    parameters = Parameterizer(
#        atom_type_names, pairs, topology.angle_atoms,
#        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
#    )
#    energies = ff.evaluate(cm, topology, parameters)
#    print(energies)
#    print(ff.b_vector)
#    ff.solver.solve(B=ff.b_vector)
#
#    def finite_difference_solve(ff, h=1e-6):
#        dsolve = torch.zeros_like(ff.last_induced_multipoles, requires_grad=False)
#        for i in range(len(dsolve)):
#            ff.last_induced_multipoles[i] += h
#            solve_plus_h = ff.solver.solve(B=ff.last_induced_multipoles)
#            ff.last_induced_multipoles[i] -= 2 * h
#            solve_minus_h = ff.solver.solve(B=ff.last_induced_multipoles)
#            ff.last_induced_multipoles[i] += h
#            dsolve += (solve_plus_h - solve_minus_h) / (2 * h)
#        return dsolve
#    
#    dsolve = finite_difference_solve(ff)
#    print(dsolve)
#    print(ff.induced_multipole_derivatives)