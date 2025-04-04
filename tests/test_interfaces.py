import torch
import os
import numpy as np

from ase.optimize import LBFGS
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import fs

from cmm.units import HARTREE2KCAL, BOHR2ANG
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM
from cmm.interfaces import CMM_ASE

from ase.units import kcal, mol, Hartree, Bohr, Angstrom

def test_ase_basic():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=False, device=device)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies = ff.evaluate(cm, topology, parameters)
    energies['tot'].backward()
    grad_1 = cm.coords.grad.detach().clone().cpu()

    ff_ase = CMM_ASE(CMM(use_ewald=True), cm, topology, parameters)
    ff_ase.calculate(ff_ase.atoms)
    print(energies)
    print(ff_ase._energies)

    forces_ref = -grad_1 * (Hartree / Bohr)
    print(forces_ref)
    print(torch.from_numpy(ff_ase.results['forces']))
    print(torch.from_numpy(ff_ase.results['forces']) - forces_ref)
    print(torch.linalg.norm((torch.from_numpy(ff_ase.results['forces']) - forces_ref), dim=1))


    assert torch.isclose(energies['tot'] * Hartree, torch.tensor(ff_ase.results['energy']))
    assert torch.allclose(forces_ref, torch.from_numpy(ff_ase.results['forces']))

def test_cmm_ase_checkpointing():
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
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
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

def test_optimize_dimers_via_ase():
    torch.set_default_dtype(torch.float64)

    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=True)
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]

    box = torch.tensor(np.eye(3) * 100, requires_grad=False)
    
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM()
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )
    ff_ase = CMM_ASE(ff, cm, topology, parameters)
    dyn = LBFGS(ff_ase.atoms)
    dyn.run(fmax=1e-6)

    # Reference CMM optimized dimer energies #
    E_w2_ref = -4.910529038105545
    E_h2o_li_ref = -34.77854035385689
    E_h2o_na_ref = -24.34707499880831
    E_h2o_k_ref = -17.623219307733585
    E_h2o_rb_ref = -15.486388340394058
    E_h2o_cs_ref = -13.83212225679549
    E_h2o_f_ref = -28.656007721154367
    E_h2o_cl_ref = -15.220650706258382
    E_h2o_br_ref = -13.235724691137875
    E_h2o_i_ref = -11.247741103845303

    # TODO: Fails because of lack of induced field gradients for FD morse.
    assert torch.isclose(torch.tensor(ff_ase.results['energy'] / (kcal / mol)), torch.tensor(E_w2_ref))

def test_optimize_water_box_via_ase():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), device=device, requires_grad=True)
    permutation = np.argsort(bonds[0], kind='stable') # Make sure sort is stable so equivalent indices don't get swapped.
    bonds[0] = bonds[0][permutation]
    bonds[1] = bonds[1][permutation]
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.get_default_dtype(), requires_grad=False, device=device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    with torch.no_grad():
        topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )
    ff_ase = CMM_ASE(ff, cm, topology, parameters)
    dyn = LBFGS(ff_ase.atoms, trajectory='water216_opt.traj')
    dyn.run(fmax=1e-3)