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

def test_ase():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True, device=device)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM()
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies = ff.evaluate(cm, topology, parameters)
    energies['tot'].backward()
    grad_1 = cm.coords.grad.detach().clone().cpu()

    ff_ase = CMM_ASE(ff, cm, topology, parameters)
    atoms = ff_ase.atoms
    atoms.get_forces()

    assert torch.isclose(energies['tot'], torch.tensor(ff_ase.results['energy']))
    assert torch.allclose(grad_1, torch.from_numpy(ff_ase.results['forces']))

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
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True, device=device)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=False)
    with torch.no_grad():
        topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )

    calculator = CMM_ASE(ff, cm, topology, parameters, output_folder=os.path.join(os.path.dirname(__file__), "scratch"))
    calculator.atoms.calc = calculator

    temperature = 300.0  # K
    MaxwellBoltzmannDistribution(calculator.atoms, temperature_K=temperature)
    Stationary(calculator.atoms)
    ZeroRotation(calculator.atoms)

    dyn = VelocityVerlet(calculator.atoms, 0.5 * fs, trajectory=os.path.join(os.path.dirname(__file__), 'scratch/w2_dynamics.traj'))
    dyn.attach(calculator.create_checkpoint, interval=5)  # Checkpoint every 5 steps
    finished = dyn.run(4)
    if finished:
        calculator.save_state(os.path.join(os.path.dirname(__file__), 'scratch/final_state.json'))

    # To restart from a checkpoint
    #calculator = CMM_ASE.load_state('checkpoint_20230320_120000.json')
    #atoms = calculator.atoms

def test_optimize_dimers_via_ase():
    torch.set_default_dtype(torch.float64)

    coords, atom_types, bonds, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=True)
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]

    box = torch.tensor(np.eye(3) * 100, dtype=torch.float64, requires_grad=False)
    
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=False)
    with torch.no_grad():
        topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )
    ff_ase = CMM_ASE(ff, cm, topology, parameters)
    ff_ase.calculate()
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
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), device=device)
    permutation = np.argsort(bonds[0], kind='stable') # Make sure sort is stable so equivalent indices don't get swapped.
    bonds[0] = bonds[0][permutation]
    bonds[1] = bonds[1][permutation]
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True, device=device)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
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