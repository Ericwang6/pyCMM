import pytest
import torch
import numpy as np
import os
import time
import torch._dynamo as dynamo

from cmm.units import HARTREE2KCAL, BOHR2ANG, BOHR2NM
from cmm.misc_utils import read_from_tinker_xyz
from cmm.cmm_water import CMMWater
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM

def test_virial_tensor():
    torch.set_default_dtype(torch.float64)

    coords, atom_types, bonds = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), requires_grad=True)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True)
    box_volume = torch.det(box)

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

    # This equation for the virial stress is derived in the Appendix of: https://doi.org/10.1016/j.cpc.2019.107057
    #right = torch.matmul(box.grad.T, box)
    #left = torch.matmul(coords.grad.T, coords)
    #virial = right + left
    #stress = virial / box_volume
    #print(virial)
    #print(stress)

def test_md():
    torch.set_default_dtype(torch.float64)

    coords, atom_types, bonds = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), requires_grad=True)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True)
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
    
    #coords = coords.detach().clone()
    #coords[0] = coords[0] + torch.tensor([0.01, 0.01, 0.01])
    #cm.update_coordinates(coords)
    #energies = ff.evaluate(cm, topology, parameters)
    #energies['tot'].backward()
    #print(coords.grad)

    #num_waters = coords.size(0) // 3
    #model = CMMWater(num_waters, do_polarization=True)
    #energies_ref = model.computeEnergy(coords, box)
    #assert torch.isclose(energies['tot'], energies_ref['tot'])

def test_optimize_nacl():
    torch.set_default_dtype(torch.float64)
    torch.autograd.set_detect_anomaly(True)

    positions = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/nacl_crystal.txt"), dtype=np.float64)
    coords = torch.tensor(positions / BOHR2NM, dtype=torch.float64, requires_grad=True)
    bonds = np.array([], dtype=np.float64)
    atom_type_names = ["" for i in range(coords.size(0))]
    for i in range(coords.size(0) // 2):
        atom_type_names[i] = "Na+"
    for i in range(coords.size(0) // 2, coords.size(0)):
        atom_type_names[i] = "Cl-"

    box = torch.tensor(np.eye(3) * 28.2 / BOHR2ANG, dtype=torch.float64, requires_grad=True)

    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 2048)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(cutoff_ewald=torch.tensor(10.0 / BOHR2ANG), ewald_tolerance=torch.tensor(1e-6), use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    ff.alpha_damp_exponent[3] = 0.0 # 3 corresponds to Cl-
    ff.alpha_damp_max[3] = 0.0
    #ff._raw_atomic_params['b_elec'][1] = 10000000000.0
    ff.rebuild_atomic_params()

    energies = ff.evaluate(cm, topology, parameters)
    energies['tot'].backward()
    #print(energies)
    #print(coords.grad)