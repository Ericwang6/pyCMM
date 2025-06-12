import torch
import os
import numpy as np

from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.md import VelocityVerlet, MDLogger
from ase.md.nptberendsen import NPTBerendsen
from ase.md.langevin import Langevin
from ase.io import Trajectory, read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import fs, bar, kB

import openmm.app as app

from cmm.interfaces import CMMCalculator
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM, BOHR2ANG


def read_xyz(xyz, device):
    coords = []
    with open(xyz) as f:
        natoms = int(f.readline().strip())
        f.readline()
        for _ in range(natoms):
            coord = [float(x) / BOHR2ANG for x in f.readline().strip().split()[1:4]]
            coords.append(coord)
    coords = torch.tensor(coords, device=device, requires_grad=True)
    return coords
    

def finite_difference(coords: torch.Tensor, f, h: float = 1e-5):
    grads_fd = torch.zeros(coords.shape)
    for i in range(coords.shape[0]):
        for w in range(coords.shape[1]):
            coords[i, w] += h
            f_plus_h = f(coords)

            coords[i, w] -= 2 * h
            f_minus_h = f(coords)
            coords[i, w] += h

            grads_fd[i, w] = (f_plus_h - f_minus_h) / (2 * h)
    return grads_fd


if __name__ == '__main__':
    device = 'cpu'
    torch.set_default_dtype(torch.float64)

    ff_path = 'tests/data/water.xml'
    pdb_path = 'tests/data/water_dimer_2.pdb'

    ff = ForceFieldXML(ff_path, device='cpu')
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)
    system = ff.parametrize(top)
    initial_coords = 10.0 * np.array(pdb.getPositions(asNumpy=True)) / BOHR2ANG
    coords = torch.from_numpy(initial_coords).requires_grad_(True)
    coords_fd = torch.from_numpy(initial_coords).requires_grad_(False)
    box = torch.tensor(np.eye(3) * 100, requires_grad=False, device=device)
    energies = system.getEnergy(coords, box)
    energies['total'].backward()
    grads_ad = coords.grad

    get_energy = lambda x :  system.getEnergy(x, box)['total']
    grads_fd = finite_difference(coords_fd, get_energy)
    print(grads_ad)
    print(grads_fd)