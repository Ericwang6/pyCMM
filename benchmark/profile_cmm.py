import torch
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cmm

from cmm.units import HARTREE2KCAL, BOHR2ANG
from cmm.misc_utils import read_xyz_tinker
from cmm.cmm_water import CMMWater
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM

def get_water_box_coords(requires_grad=True):
    labels, atom_types, coords, bonds = read_xyz_tinker(os.path.join(os.path.dirname(__file__), "../tests/data/water_216.xyz"))
    permutation = np.argsort(bonds[0], kind='stable') # Make sure sort is stable so equivalent indices don't get swapped.
    bonds[0] = bonds[0][permutation]
    bonds[1] = bonds[1][permutation]
    atom_types = torch.tensor(atom_types, dtype=torch.long) - 1
    coords = torch.tensor(coords / BOHR2ANG, dtype=torch.float64, requires_grad=requires_grad)
    return coords, atom_types, bonds

def profile_cmm_evaluation():
    torch.set_default_dtype(torch.float64)

    coords, atom_types, bonds = get_water_box_coords(requires_grad=True)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM()
    ff_opt = torch.compile(ff)

    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff_opt.atomic_params, ff_opt.pair_params, ff_opt.pair_pair_params, ff_opt.pair_angle_params, ff_opt.angle_params
    )
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("CMM_evaluate_water_box_216_1"):
            energies = ff_opt.evaluate(cm, topology, parameters)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("CMM_evaluate_water_box_216_2"):
            energies = ff_opt.evaluate(cm, topology, parameters)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

if __name__ == "__main__":
    profile_cmm_evaluation()
    