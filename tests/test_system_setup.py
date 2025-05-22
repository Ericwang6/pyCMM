import torch
import os
import numpy as np

from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.settings import Settings, MolecularDynamicsSettings, NeighborListSettings
from cmm.misc_utils import read_from_tinker_xyz
from cmm.parameters import Parameterizer
from cmm.system import System, create_system_from_ext_xyz_file
from cmm.force_fields.spcfw import SPCFW
from cmm.force_fields.cmm import CMM2
from cmm.units import BOHR2ANG, HARTREE2KCAL
from cmm.force_field import CMM

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
    settings = Settings()
    settings.add_neighbor_list_settings(padding=1.5)
    settings.add_long_range_electrostatics_settings(cutoff=9.0, tolerance=1e-10)
    settings.add("md", md_settings)

    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True, device=device)
    
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)

    topology = Topology(bonds, coords.size(0), device)
    system = System(coords, box, atom_type_names, topology, settings)
    ff = SPCFW(system)
    ff.forward(system)
    print(ff.energies)
    ff.energies['V_total'].backward()
    print(coords.grad)
    print(box.grad)
    
def test_cmm_water_box_setup():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    settings = Settings()
    settings.add_neighbor_list_settings(padding=1.5)
    settings.add_long_range_electrostatics_settings(cutoff=9.0, tolerance=1e-10)

    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True, device=device)
    
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)

    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM()
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    torch.set_printoptions(9)
    energies = ff.evaluate(cm, topology, parameters)
    energies['total'].backward()
    print(energies)


    system = System(coords, box, atom_type_names, topology, settings)
    ff = CMM2(system)
    ff.forward(system)
    print(ff.energies)
    ff.energies['V_total'].backward()
    print(coords.grad)
    print(box.grad)