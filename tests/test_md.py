import pytest
import torch
import numpy as np
import os

from cmm.units import HARTREE2KCAL, BOHR2ANG
from cmm.misc_utils import read_xyz_tinker
from cmm.cmm_water import CMMWater
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology

def get_water_box_coords(requires_grad=True):
    labels, atom_types, coords, bonds = read_xyz_tinker(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"))
    permutation = np.argsort(bonds[0])
    bonds[0] = bonds[0][permutation]
    bonds[1] = bonds[1][permutation]
    coords = coords[[permutation]]
    coords = torch.tensor(coords / BOHR2ANG, dtype=torch.float64, requires_grad=requires_grad)
    return coords, atom_types, bonds

def test_md():
    torch.set_default_dtype(torch.float64)
    coords, atom_types, bonds = get_water_box_coords(requires_grad=True)
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True)
    topology = Topology(bonds)

    #cm = CoordinateManager(coords, box, 12.0)
    #pairs, dists, distance_vectors = cm.get_intermolecular_distances_vectors_and_pairs()
    #print(dists)
    #print(distance_vectors)
    # cm.get_dists_and_vecs(i)
    # cm.get_bond_lengths(i)
    # cm.get_angles(i)
    # etc.

    #num_waters = coords.size(0) // 3
    #model = CMMWater(num_waters, do_polarization=True)
    #energies = model.computeEnergy(coords, box)
    