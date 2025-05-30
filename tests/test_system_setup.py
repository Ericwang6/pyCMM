import torch
import os
import numpy as np

from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.settings import Settings, MolecularDynamicsSettings, NeighborListSettings, PolarizationSettings, ShortRangeSettings
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
    
    settings = Settings()
    settings.add_neighbor_list_settings(cutoff=30.0, padding=1.5)
    settings.add_long_range_electrostatics_settings(use_long_range=False)
    settings.add_long_range_dispersion_settings(use_long_range=False)
    settings.add("polarization", PolarizationSettings())
    settings.add("short_range", ShortRangeSettings())

    systems = create_system_from_ext_xyz_file(os.path.join(os.path.dirname(__file__), "data/water_clusters.xyz"), settings, requires_grad=True, device=device)
    ff = CMM2(systems[0])
    ff.forward(systems[0])
    V_total = ff.energies['V_total']
    ff.forward(systems[0])
    V_total = V_total + ff.energies['V_total']
    V_total.backward()


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
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True, requires_box_grad=True, device=device)
    
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
    
def test_cmm_water_box_against_original_implementation():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True, device=device)
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)
    topology = Topology(bonds, coords.size(0), device)

    reference_energies = {
        'perm_elec': torch.tensor(-1.530321435),
        'pol': torch.tensor(-1.066418481),
        'ct_direct': torch.tensor(-1.463103904),
        'xpol': torch.tensor(-0.238940206),
        'pauli': torch.tensor(6.577868084),
        'disp': torch.tensor(-2.070634588),
        'deformation': torch.tensor(0.370158548),
        'bond': torch.tensor(0.229082034),
        'angle': torch.tensor(0.133542917),
        'bond_bond': torch.tensor(-0.000843700),
        'bond_angle': torch.tensor(0.008377297),
        'total': torch.tensor(-3.373368613),
        'ewald': torch.tensor(-3.916683062),
        'disp_lr': torch.tensor(-0.035293569),
    }

    settings = Settings()
    settings.add_neighbor_list_settings(padding=1.5)
    settings.add_long_range_electrostatics_settings(cutoff=9.0, tolerance=1e-6)
    settings.add("polarization", PolarizationSettings())
    settings.add("short_range", ShortRangeSettings())

    system = System(coords, box, atom_type_names, topology, settings)
    ff = CMM2(system)
    ff.forward(system)
    ff.energies['V_total'].backward()
    assert torch.isclose(ff.energies['V_total'], reference_energies['total'])