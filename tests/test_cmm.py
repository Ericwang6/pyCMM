import pytest

import itertools
from pprint import pprint

import torch
from torch_scatter import scatter
import numpy as np

import os

from cmm.units import BOHR2NM, HARTREE2KJ, HARTREE2KCAL, BOHR2ANG, DEBYE2EA
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM

def finite_difference(coords: torch.Tensor, f, h: float = 1e-5):
    grads_fd = torch.zeros(coords.shape)
    for i in range(coords.shape[0]):
        for w in range(coords.shape[1]):
            coords[i, w] += h
            f_plus_h = f(coords)

            coords[i, w] -= 2 * h
            f_minus_h = f(coords)
            coords[i, w] += h

            grads_fd[i, w] = (f_plus_h - f_minus_h) / (2 * h)
    return grads_fd

def test_total_energy_and_total_gradients_ion_ion():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    coords_no_grad, _, _, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/na_cl.xyz"), requires_grad=False, device=device)
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/na_cl.xyz"), requires_grad=True, device=device)
    atom_indices_to_names = {
        0: "O_water", 1: "H_water",
        2: "F-", 3: "Cl-", 4: "Br-", 5: "I-",
        6: "Li+", 7: "Na+", 8: "K+", 9: "Rb+", 10: "Cs+",
        11: "Mg2+", 12: "Ca2+"
    }
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 100, requires_grad=False, device=device)
    
    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=False, use_lr_dispersion=False)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )
    energies_ff = ff.evaluate(cm, topology, parameters)
    total_ref = torch.tensor([-132.66013773479762 / HARTREE2KCAL])
    assert torch.allclose(energies_ff['total'].cpu(), total_ref)

def test_total_energy_and_total_gradients_ion_water_cluster():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    coords_no_grad, _, _, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/w4_na_cl.xyz"), requires_grad=False, device=device)
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/w4_na_cl.xyz"), requires_grad=True, device=device)
    atom_indices_to_names = {
        0: "O_water", 1: "H_water",
        2: "F-", 3: "Cl-", 4: "Br-", 5: "I-",
        6: "Li+", 7: "Na+", 8: "K+", 9: "Rb+", 10: "Cs+"
    }
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 100, requires_grad=False, device=device)
    
    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(cutoff_short_range=torch.tensor(15.0 / BOHR2ANG), use_ewald=False, use_lr_dispersion=False, solve_tolerance=torch.tensor(1e-10))
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies_ff = ff.evaluate(cm, topology, parameters)
    total_ref = torch.tensor([-192.01391054587654])
    # ^^^ The induced bond polarization part is 0.03793419591828556
    # but that doesn't work yet cause I don't have induced dipole derivatives yet.
    # Note the above reference excludes the induced bond polarization part.
    # To be extra clear that is the contribution from just the induced fields
    # on the total deformation energy.
    total_energy = energies_ff['total'].cpu() * HARTREE2KCAL
    assert torch.allclose(total_energy, total_ref, atol=0.01)
    # ^^^ This difference comes from slightly different unit conversions between the two codes.

    energies_ff['total'].backward()
    grads_ad_1 = coords.grad.clone()

    def get_total_energy(coords: torch.Tensor):
        topology = Topology(bonds, coords.size(0), device)
        cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
        pairs, _, _ = cm.get_distances_vectors_and_pairs()
        ff = CMM(cutoff_short_range=torch.tensor(15.0 / BOHR2ANG), use_ewald=False, use_lr_dispersion=False,)
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )
        energies_ff = ff.evaluate(cm, topology, parameters)
        total_energy = energies_ff['total']
        return total_energy

    grads_fd_1 = finite_difference(coords_no_grad, get_total_energy, h=1e-5).to(device)
    if not torch.allclose(grads_ad_1, grads_fd_1):
        print(grads_ad_1 - grads_fd_1)
    assert torch.allclose(grads_ad_1, grads_fd_1)

def test_total_energy_and_total_gradients_ion_water():
    torch.set_default_dtype(torch.float64)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords_no_grad, _, _, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/h2o_f.xyz"), requires_grad=False, device=device)
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/h2o_f.xyz"), requires_grad=True, device=device)
    
    #coords_no_grad, _, _, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/w5_f.xyz"), requires_grad=False)
    #coords, atom_types, bonds, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/w5_f.xyz"), requires_grad=True)
    atom_indices_to_names = {
        0: "O_water", 1: "H_water",
        2: "F-", 3: "Cl-", 4: "Br-", 5: "I-",
        6: "Li+", 7: "Na+", 8: "K+", 9: "Rb+", 10: "Cs+"
    }
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 100, requires_grad=False, device=device)
    
    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=False, use_lr_dispersion=False, cutoff_short_range=torch.tensor(10.0 / BOHR2ANG))
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies_ff = ff.evaluate(cm, topology, parameters)
    total_ref = torch.tensor([-28.231218043296266], device=device)
    total_energy = energies_ff['total'] * HARTREE2KCAL
    assert torch.allclose(total_energy, total_ref)

    energies_ff['total'].backward()
    grads_ad_1 = coords.grad.clone()

    def get_total_energy(coords: torch.Tensor):
        topology = Topology(bonds, coords.size(0), device)
        cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
        pairs, _, _ = cm.get_distances_vectors_and_pairs()
        ff = CMM(use_ewald=False, use_lr_dispersion=False, cutoff_short_range=torch.tensor(10.0 / BOHR2ANG))
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )
        energies_ff = ff.evaluate(cm, topology, parameters)
        total_energy = energies_ff['total']
        return total_energy

    grads_fd_1 = finite_difference(coords_no_grad, get_total_energy, h=1e-5).to(device)
    if not torch.allclose(grads_ad_1, grads_fd_1):
        print(grads_ad_1 - grads_fd_1)
    assert torch.allclose(grads_ad_1, grads_fd_1)

def test_total_energy_and_total_gradients():
    torch.set_default_dtype(torch.float64)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords_no_grad, _, _, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=False, device=device)
    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=True, device=device)
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 100, requires_grad=True, device=device)

    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=False, use_lr_dispersion=False)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies_ff = ff.evaluate(cm, topology, parameters)
    total_ref = torch.tensor([-4.768231511534177])
    total_ff = energies_ff['total'].cpu() * HARTREE2KCAL
    assert torch.allclose(total_ff, total_ref)

    energies_ff['total'].backward()
    grads_ad_1 = coords.grad.clone().cpu()

    def get_total_energy(coords: torch.Tensor):
        topology = Topology(bonds, coords.size(0), device)
        cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
        pairs, _, _ = cm.get_distances_vectors_and_pairs()
        ff = CMM(use_ewald=False, use_lr_dispersion=False)
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
        )
        energies_ff = ff.evaluate(cm, topology, parameters)
        total_energy = energies_ff['total'].cpu()
        return total_energy

    grads_fd_1 = finite_difference(coords_no_grad, get_total_energy, h=1e-5)
    if not torch.allclose(grads_ad_1, grads_fd_1):
        print(grads_ad_1 - grads_fd_1)

    assert torch.allclose(grads_ad_1, grads_fd_1)

def test_dipole_moment():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    coords, atom_types, bonds, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=True)
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 100, requires_grad=True)
    
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(solve_tolerance=1e-15)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies_ff = ff.evaluate(cm, topology, parameters)
    dipole_moment = ff.get_dipole_moment(cm.coords, box)
    assert torch.allclose(dipole_moment, torch.tensor([ -0.0198134931, -1.05513346, 0.00294907]))

    coords, atom_types, bonds, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), requires_grad=True)
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, device=device)
    
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, 1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True, use_lr_dispersion=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies_ff = ff.evaluate(cm, topology, parameters)
    dipole_moment = ff.get_dipole_moment(cm.coords, box)
    print(dipole_moment)

def test_nonbonded_interactions():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=False, device=device)
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 100, requires_grad=False, device=device)
    
    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=False, use_lr_dispersion=False)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )
    energies = ff.evaluate(cm, topology, parameters)

    torch.set_printoptions(9)

    ct_direct = energies['ct_direct'].cpu() * HARTREE2KCAL
    perm_elec = energies['perm_elec'].cpu() * HARTREE2KCAL
    pol_ct = energies['pol'].cpu() * HARTREE2KCAL
    pauli = energies['pauli'].cpu() * HARTREE2KCAL
    disp = energies['disp'].cpu() * HARTREE2KCAL
    xpol = energies['xpol'].cpu() * HARTREE2KCAL

    ct_direct_ref = torch.tensor([-2.777994825946152])
    xpol_ref = torch.tensor([-0.4036285834810728])
    perm_elec_ref = torch.tensor([-9.705778546689396])
    pol_ct_ref = torch.tensor([-0.56935496823574])
    pauli_ref = torch.tensor([10.70659662118688])
    disp_ref = torch.tensor([-2.054389376337415])
    assert torch.allclose(ct_direct, ct_direct_ref)
    assert torch.allclose(xpol, xpol_ref)
    assert torch.allclose(disp, disp_ref)
    assert torch.allclose(pauli, pauli_ref)
    assert torch.allclose(perm_elec, perm_elec_ref)
    assert torch.allclose(pol_ct, pol_ct_ref)

