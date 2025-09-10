import os
import torch
torch.set_printoptions(precision=8)
import openmm.app as app
import cmm.units as units
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from pprint import pprint


def test_batched_system_water_dimer():
    device = 'cpu'
    float_dtype = torch.float64
    torch.set_default_dtype(float_dtype)

    pdb = app.PDBFile(os.path.join(os.path.dirname(__file__), 'data/water_dimer.pdb'))
    coords_list = (pdb.getPositions(asNumpy=True)._value / units.BOHR2NM).tolist() 
    coords_bz = torch.tensor([coords_list, coords_list], device=device, dtype=float_dtype)

    coords = torch.tensor((pdb.getPositions(asNumpy=True)._value / units.BOHR2NM).tolist(), device=device, dtype=float_dtype)
    box = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]], device=device, dtype=float_dtype)

    ff = ForceFieldXML(os.path.join(os.path.dirname(__file__), 'data/water.xml'), device=device, float_dtype=torch.float64)
    top = Topology.fromOpenmm(pdb.topology, device)
    bs = ff.parametrize(top, batch=True, use_hardness_change=False, use_fd_morse=False)
    
    energies = bs.getEnergy(coords_bz, energy_in_kcal=True)
    pprint(energies)

#     pprint({k: v * units.HARTREE2KCAL for k, v in ff.parametrize(top, batch=False, use_hardness_change=False, use_fd_morse=False).getEnergy(coords, box).items()})


# def test_batched_system_methanol_dimer():
#     device = 'cpu'
#     float_dtype = torch.float64
#     torch.set_default_dtype(float_dtype)

#     pdb = app.PDBFile(os.path.join(os.path.dirname(__file__), 'data/methanol_dimer.pdb'))
#     coords_list = (pdb.getPositions(asNumpy=True)._value / units.BOHR2NM).tolist() 
#     coords_bz = torch.tensor([coords_list, coords_list], device=device, dtype=float_dtype)

#     coords = torch.tensor((pdb.getPositions(asNumpy=True)._value / units.BOHR2NM).tolist(), device=device, dtype=float_dtype)
#     box = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]], device=device, dtype=float_dtype)

#     ff = ForceFieldXML(os.path.join(os.path.dirname(__file__), 'data/water_methanol.xml'), device=device, float_dtype=torch.float64)
#     top = Topology.fromOpenmm(pdb.topology, device)
#     bs = ff.parametrize(top, batch=True, use_hardness_change=False, use_fd_morse=False)
    
#     energies = bs.getEnergy(coords_bz, energy_in_kcal=True)
#     pprint(energies)

#     pprint({k: v * units.HARTREE2KCAL for k, v in ff.parametrize(top, batch=False, use_hardness_change=False, use_fd_morse=False).getEnergy(coords, box).items()})