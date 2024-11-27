import pytest
import torch
import numpy as np
import os

from cmm.units import HARTREE2KCAL, BOHR2ANG
from cmm.misc_utils import read_xyz_tinker
from cmm.cmm_water import CMMWater
from cmm.coordinate_manager import CoordinateManager

def get_water_box_coords(requires_grad=True):
    labels, coords = read_xyz_tinker(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"))
    coords = torch.tensor(coords / BOHR2ANG, dtype=torch.float64, requires_grad=requires_grad)
    return coords

def test_md():
    torch.set_default_dtype(torch.float64)
    coords = get_water_box_coords(requires_grad=True)
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True)
    
    cm = CoordinateManager(coords, box, 12.0)
    pairs, dists, distance_vectors = cm.get_intermolecular_distances_vectors_and_pairs()
    print(dists)
    print(distance_vectors)
    # cm.get_dists_and_vecs(i)
    # cm.get_bond_lengths(i)
    # cm.get_angles(i)
    # etc.

    #num_waters = coords.size(0) // 3
    #model = CMMWater(num_waters, do_polarization=True)
    #energies = model.computeEnergy(coords, box)
    