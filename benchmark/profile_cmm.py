import torch

import numpy as np
import os

from cmm.units import HARTREE2KCAL, BOHR2ANG, AMU2ELECTRON_MASS, FS2AU, KB_EhPerK
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM
from cmm.spcfw import SPCfw
from cmm.interfaces import CMM_ASE
from cmm.spcfw_interface import SPCfw_ASE
from cmm.timing_context import *

from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.units import bar

def profile_spcfw_md():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "../tests/data/water_216.xyz"), device=device, requires_grad=True)
    permutation = np.argsort(bonds[0], kind='stable') # Make sure sort is stable so equivalent indices don't get swapped.
    bonds[0] = bonds[0][permutation]
    bonds[1] = bonds[1][permutation]
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.get_default_dtype(), requires_grad=True, device=device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = SPCfw()
    with torch.no_grad():
        topology = Topology(bonds, cm.neighbor_list, coords.size(0))
        parameters = Parameterizer(
            atom_type_names, pairs, topology.angle_atoms,
            ff.atomic_params, ff.pair_params, {}, {}, ff.angle_params
        )
    ff_ase = SPCfw_ASE(ff, cm, topology, parameters)
    fcf = FrechetCellFilter(ff_ase.atoms, hydrostatic_strain=True, scalar_pressure=1.01325 * bar)
    opt = LBFGS(fcf, trajectory='water216_cell_opt_spcfw.traj')
    opt.run(fmax=1e-2, steps=20)
    ff_ase.save_state("water216_cell_opt_spcfw.json")
    print_timing_stats()

if __name__ == "__main__":
    profile_spcfw_md()
    