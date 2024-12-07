import pytest
import torch
import numpy as np
import os

from cmm.units import HARTREE2KCAL, BOHR2ANG
from cmm.misc_utils import read_xyz_tinker
from cmm.cmm_water import CMMWater
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM

def get_water_box_coords(requires_grad=True):
    labels, atom_types, coords, bonds = read_xyz_tinker(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"))
    permutation = np.argsort(bonds[0], kind='stable') # Make sure sort is stable so equivalent indices don't get swapped.
    bonds[0] = bonds[0][permutation]
    bonds[1] = bonds[1][permutation]
    atom_types = torch.tensor(atom_types, dtype=torch.long) - 1
    coords = torch.tensor(coords / BOHR2ANG, dtype=torch.float64, requires_grad=requires_grad)
    return coords, atom_types, bonds

def test_md():
    torch.set_default_dtype(torch.float64)

    coords, atom_types, bonds = get_water_box_coords(requires_grad=True)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True)
    cm = CoordinateManager(coords, box, 12.0)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM()
    parameters = Parameterizer(atom_type_names, ff._raw_atomic_params)
    
    #ff.evaluate(cm, topology, parameters)
    
    #print(pairs[topology.angle_pairs])

    #pairs, dists, distance_vectors = cm.get_intermolecular_distances_vectors_and_pairs()
    #print(dists)
    #print(distance_vectors)

    #num_waters = coords.size(0) // 3
    #model = CMMWater(num_waters, do_polarization=True)
    #energies = model.computeEnergy(coords, box)
    #print(energies)