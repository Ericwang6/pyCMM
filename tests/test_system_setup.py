import torch
import os
import numpy as np

from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from ase.units import bar

from cmm.topology import Topology
from cmm.settings import Settings, MolecularDynamicsSettings, PolarizationSettings, ShortRangeSettings
from cmm.misc_utils import read_from_tinker_xyz
from cmm.system import System, create_system_from_ext_xyz_file, create_system_from_xyz_file
from cmm.force_fields.spcfw import SPCFW
from cmm.force_fields.cmm import CMM2
from cmm.units import BOHR2ANG, HARTREE2KCAL
from cmm.ase_interface import ASE_Interface

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
    
def test_cmm_ice_box():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    system_file = os.path.join(os.path.dirname(__file__), "data/ice_Ih_3x3_cell.xyz")

    settings = Settings()
    settings.add_neighbor_list_settings(padding=1.5)
    settings.add_long_range_electrostatics_settings(cutoff=9.0, tolerance=1e-6)
    settings.add("polarization", PolarizationSettings())
    settings.add("short_range", ShortRangeSettings())

    # 90.0, 90.0, 60.0
    ice_box = torch.from_numpy(np.array([
        [23.34,  0.  ,  0.  ],
        [11.67      , 20.21303292,  0.        ],
        [ 0.  ,  0.  , 21.99]]
    ))
    ice_systems = create_system_from_xyz_file(system_file, settings, requires_grad=True, requires_box_grad=True, device=device, box=ice_box)

    system = ice_systems[0]
    ff = CMM2(system)
    ff.forward(system)
    torch.set_printoptions(9)
    for key in ff.energies.keys():
        print(key, " ", ff.energies[key] * HARTREE2KCAL)

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

def test_cmm_on_reference_clusters():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    settings = Settings()
    settings.add_neighbor_list_settings(cutoff=30.0, padding=1.5)
    settings.add_long_range_electrostatics_settings(use_long_range=False, cutoff=30.0)
    settings.add_long_range_dispersion_settings(use_long_range=False)
    settings.add("polarization", PolarizationSettings(tolerance=1e-10))
    settings.add("short_range", ShortRangeSettings(cutoff=15.0))

    ref_cluster_systems = create_system_from_xyz_file("/home/heindelj/OneDrive/Documents/Coding_Projects/python_development/pyCMM/tests/data/water_clusters.xyz", settings, requires_grad=True, device=device)
    system = ref_cluster_systems[-1]
    ff = CMM2(system)
    ff.forward(system)
    ff.energies['V_total'].backward()
    torch.set_printoptions(9)
    for key in ff.energies.keys():
        print(key, " ", ff.energies[key] * HARTREE2KCAL)

def test_cmm_md_on_water_box():
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    settings = Settings()
    settings.add_neighbor_list_settings(cutoff=9.0, padding=1.5)
    settings.add_long_range_electrostatics_settings(use_long_range=True, cutoff=9.0)
    settings.add_long_range_dispersion_settings(use_long_range=True)
    settings.add("polarization", PolarizationSettings(tolerance=1e-6))
    settings.add("short_range", ShortRangeSettings(cutoff=6.0))

    water_box_system = create_system_from_ext_xyz_file(os.path.join(os.path.dirname(__file__), "data/water_216_ext.xyz"), settings, requires_grad=True, device=device)
    system = water_box_system[0]
    ff = CMM2(system)
    ff.forward(system)
    ff.energies['V_total'].backward()
    import copy
    initial_energies = copy.copy(ff.energies)
    
    system_2 = System.from_instance(system)
    ff_2 = CMM2(system_2)
    calculator = ASE_Interface(ff_2, system_2)
    calculator.calculate()
    fcf = FrechetCellFilter(calculator.atoms, hydrostatic_strain=True, scalar_pressure=1.01325 * bar)
    opt = LBFGS(fcf, trajectory='water216_cell_opt.traj')
    opt.run(steps=2)
    calculator.atoms.write("water216_cell_opt.xyz")