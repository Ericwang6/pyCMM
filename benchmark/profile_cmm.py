import torch
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cmm

from cmm.units import HARTREE2KCAL, BOHR2ANG
from cmm.misc_utils import read_xyz_tinker
from cmm.cmm_water import CMMWater
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM

def get_water_box_coords(requires_grad=True, device="cpu"):
    labels, atom_types, coords, bonds = read_xyz_tinker(os.path.join(os.path.dirname(__file__), "../tests/data/water_216.xyz"))
    permutation = np.argsort(bonds[0], kind='stable') # Make sure sort is stable so equivalent indices don't get swapped.
    bonds[0] = bonds[0][permutation]
    bonds[1] = bonds[1][permutation]
    atom_types = torch.tensor(atom_types, dtype=torch.long, requires_grad=False, device=device) - 1
    coords = torch.tensor(coords / BOHR2ANG, dtype=torch.float64, requires_grad=requires_grad, device=device)
    return coords, atom_types, bonds

def profile_cmm_evaluation():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords, atom_types, bonds = get_water_box_coords(requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True, device=device)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM()

    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as prof:
        with record_function("CMM_evaluate_water_box_216"):
            energies = ff.evaluate(cm, topology, parameters)
            energies['tot'].backward()
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))

if __name__ == "__main__":
    profile_cmm_evaluation()
    