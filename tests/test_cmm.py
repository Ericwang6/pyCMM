import pytest
import os
from tqdm import tqdm
from pprint import pprint as pp
import torch
import openmm.app as app

os.environ["TORCH_COMPILE_DISABLE"] = "1"
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM
torch.set_printoptions(precision=8)
torch.set_default_dtype(torch.float64)


def finite_difference(f, x: torch.Tensor, h=0.0001, *args, **kwargs):
    tmp = x.requires_grad
    x.requires_grad = False
    with torch.no_grad():
        grad = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        for idx in tqdm(torch.cartesian_prod(*(torch.arange(n) for n in x.shape)), desc='Compute Finite Difference'):
            x[*idx] += h
            v0 = f(x, *args, **kwargs)
            x[*idx] -= 2 * h
            v1 = f(x, *args, **kwargs)
            x[*idx] += h
            grad[*idx] = (v0 - v1) / (2 * h)
    x.requires_grad = tmp
    return grad


def test_forces():
    ff_path = '/pscratch/sd/e/eric6/pycmm-dev/workspace/water_mc/water_refit.xml'
    pdb_path = os.path.join(os.path.dirname(__file__), 'data/water_216.pdb')

    # ff_path = '/pscratch/sd/e/eric6/pycmm-dev/optimization/cmm_ethane.xml'
    # pdb_path = '/pscratch/sd/e/eric6/pycmm-dev/workspace_new/ethane_216_gaff_opt.pdb'
    # pdb_path = '/pscratch/sd/e/eric6/pycmm-dev/workspace_new/methanol_216_gaff_opt.pdb'
    
    device = 'cuda'
    ff = ForceFieldXML(ff_path, device=device)
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)

    coords = torch.tensor(pdb.getPositions(asNumpy=True)._value / BOHR2NM, device=device, requires_grad=True)
    box = torch.tensor([[vec.x / BOHR2NM, vec.y / BOHR2NM, vec.z / BOHR2NM] for vec in pdb.topology.getPeriodicBoxVectors()], device=device, requires_grad=True)

    system_ref = ff.parametrize(top, use_fd_morse=True, use_polarization=True, polarization_tolerance=1e-7, use_hardness_change=False, cutoff_sr=9.0, use_switch=True, use_customized_ops=False)
    energies_ref = system_ref.getEnergy(coords, box)
    energies_ref['total'].backward()
    coords_grad_ad = coords.grad.clone()
    coords.grad = None

    energy_fn = lambda x: system_ref.getEnergy(x, box)['total']

    coords_grad_fd = finite_difference(energy_fn, coords)

    print(coords_grad_ad[:3], coords_grad_fd[:3])