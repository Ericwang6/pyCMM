import torch
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import numpy as np

import openmm.app as app
import sys
sys.path.insert(0,"/pscratch/sd/a/asa/software/pyCMM")
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM
torch.set_printoptions(precision=10)
torch.set_default_dtype(torch.float64)

from pprint import pprint as pp


def test_customized_ops():
    ff_path ="water-aalim-angd.xml"
    #ff_path = "nofd.xml"
    pdb_path = os.path.join(os.path.dirname(__file__), 'data/water_216.pdb')
    
    device = 'cuda'
    ff = ForceFieldXML(ff_path, device=device)
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)

    coords = torch.tensor(pdb.getPositions(asNumpy=True)._value / BOHR2NM, device=device, requires_grad=True)
    box = torch.tensor([[vec.x / BOHR2NM, vec.y / BOHR2NM, vec.z / BOHR2NM] for vec in pdb.topology.getPeriodicBoxVectors()], device=device, requires_grad=True)

    system_ref = ff.parametrize(top, use_fd_morse=True, use_fd_angle=True, use_polarization=True, polarization_tolerance=1e-7, use_hardness_change=False, cutoff_sr=9.0, use_switch=True, use_customized_ops=False)
    energies_ref = system_ref.getEnergy(coords, box)
    energies_ref['total'].backward()
    coords_grad_ref = coords.grad.numpy(force=True)
    coords.grad = None
    print('\n')
    print('===== No Customized Ops =====')
    pp(energies_ref)
    pp(coords_grad_ref[:3])
    
    system = ff.parametrize(top, use_fd_morse=True, use_fd_angle=True, use_polarization=True, polarization_tolerance=1e-7, use_hardness_change=False, cutoff_sr=9.0, use_customized_ops=True, use_switch=True)
    energies = system.getEnergy(coords, box)
    energies['total'].backward()
    coords_grad = coords.grad.numpy(force=True)
    print('===== Customized Ops =====')
    pp(energies)
    pp(coords_grad[:3])
