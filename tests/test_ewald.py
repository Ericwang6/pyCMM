import pytest
import torch
import numpy as np
import os
import time
import torch._dynamo as dynamo

from cmm.units import HARTREE2KCAL, BOHR2ANG
from cmm.misc_utils import read_from_tinker_xyz
from cmm.cmm_water import CMMWater
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM

def test_direct_ewald():
    torch.set_default_dtype(torch.float64)

    coords_no_grad, _, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=False)
    coords, atom_types, bonds = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=True)
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 100, dtype=torch.float64, requires_grad=True)
    
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )
    energies_ff = ff.evaluate(cm, topology, parameters)
    #total_ref = torch.tensor([-4.768231511534177 / HARTREE2KCAL])
    #assert torch.allclose(energies_ff['tot'], total_ref)

    #def get_total_energy(coords: torch.Tensor):
    #    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    #    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    #    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    #    ff = CMM()
    #    parameters = Parameterizer(
    #        atom_type_names, pairs, topology.angle_atoms,
    #        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    #    )
    #    energies_ff = ff.evaluate(cm, topology, parameters)
    #    total_energy = energies_ff['tot']
    #    return total_energy
    #
    #grads_fd_1 = finite_difference(coords_no_grad, get_total_energy, h=1e-5)
    #energies_ff['tot'].backward(retain_graph=True)
    #grads_ad_1 = coords.grad.clone()
    #if not torch.allclose(grads_ad_1, grads_fd_1):
    #    print(grads_ad_1 - grads_fd_1)
    #
    #assert torch.allclose(grads_ad_1, grads_fd_1)