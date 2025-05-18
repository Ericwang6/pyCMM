import torch
import os
import numpy as np

from cmm.system import System, create_system_from_ext_xyz_file
from cmm.force_fields.spcfw import SPCFW
from cmm.topology import Topology
from cmm.settings import Settings, MolecularDynamicsSettings, NeighborListSettings
from cmm.misc_utils import read_from_tinker_xyz
from cmm.units import BOHR2ANG, HARTREE2KCAL

def test_system_setup_from_file():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    md_settings = MolecularDynamicsSettings(
        ensemble="NVT", timestep=0.5,
        n_steps=50, temperature=298.15
    )
    nl_settings = NeighborListSettings(
        padding=1.5
    )
    settings = Settings()
    settings.add_neighbor_list(nl_settings)
    settings.add("md", md_settings)

    create_system_from_ext_xyz_file(os.path.join(os.path.dirname(__file__), "data/water_clusters.xyz"), requires_grad=True, device=device)
    # HERE: Keep working on the system and settings stuff.
    # 1) Load system from extxyz file
    # 2) Make system have same functionality as coordinate manager

def test_spcfw_water_box_setup():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    md_settings = MolecularDynamicsSettings(
        ensemble="NVT", timestep=0.5,
        n_steps=50, temperature=298.15
    )
    nl_settings = NeighborListSettings(
        padding=1.5
    )
    settings = Settings()
    settings.add_neighbor_list(nl_settings)
    settings.add("md", md_settings)

    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True, device=device)
    
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)

    topology = Topology(bonds, coords.size(0), device)
    system = System(coords, box, atom_type_names, topology, settings)
    ff = SPCFW()
    ff.forward(system)
    ff.energies['V_bond'].backward()
    
