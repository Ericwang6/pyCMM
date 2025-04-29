import pytest

import itertools
from pprint import pprint

import torch
from torch_scatter import scatter
import numpy as np

import os

from cmm.units import BOHR2NM, HARTREE2KJ, HARTREE2KCAL, BOHR2ANG, DEBYE2EA
from cmm.multipole import computeLocal2GlobalRotationMatrix, rotateMultipoles, rotateQuadrupoles, computeCartesianQuadrupoles
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM

def test_lennard_jones():
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
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True, use_lr_dispersion=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    ff_ase = CMM_ASE(ff, cm, topology, parameters)

    virial_fd = torch.from_numpy(calculate_virial_finite_difference(ff_ase.atoms, epsilon=1e-6) / HARTREE2EV).to(device)
    stress_fd = virial_fd / box_volume

    print(stress)
    print(stress_fd)

    assert torch.allclose(stress_fd, stress)