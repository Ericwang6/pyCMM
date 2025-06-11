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
    

def create_all_logger(filename: str, atoms, dynamics):
    """
    Create a logging function for MD simulations that writes a header to the given file and
    returns a callable `log_all()` to record step, time (fs), temperature (K),
    total energy (eV), and density (g/cm3) at each interval.
    """
    # Open the log file and write the header line
    f = open(filename, 'w')
    header = '# step    time(fs)    temperature(K)    total_energy(eV)    density(g/cm3)'
    f.write(header+'\n')
    print(header)

    def log_all():
        # Get current step number
        step = dynamics.get_number_of_steps()
        # Get current simulation time in femtoseconds
        time_fs = dynamics.get_time() / fs

        # Compute temperature: T = 2 * E_kinetic / (3 * N * k_B)
        E_kin = atoms.get_kinetic_energy()
        N = atoms.get_number_of_atoms()
        temperature = 2 * E_kin / (3 * N * kB)

        # Compute total energy: potential + kinetic
        E_pot = atoms.get_potential_energy()
        total_energy = E_kin + E_pot

        # Compute density: mass (amu) to grams, volume in Å^3 to cm^3
        total_mass_amu = sum(atoms.get_masses())
        volume_A3 = atoms.get_volume()
        # 1 amu = 1.66054e-24 g; 1 Å^3 = 1e-24 cm^3
        density = total_mass_amu * 1.66054e-24 / (volume_A3 * 1e-24)

        # Write a line of data to the log file
        msg = f'{step:6d}  {time_fs:8.2f}  {temperature:10.2f}  {total_energy:12.6f}  {density:12.6f}'
        f.write(msg+'\n')
        print(msg)

    return log_all


if __name__ == '__main__':
    device = 'cuda'
    torch.set_default_dtype(torch.float64)

    ff_path = 'tests/data/water.xml'
    pdb_path = 'tests/data/water_216.pdb'

    ff = ForceFieldXML(ff_path, device='cuda')
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)
    system = ff.parametrize(top)

    # coords = torch.tensor(pdb.getPositions(asNumpy=True)._value / BOHR2NM, device=device, requires_grad=True)
    # coords = read_xyz('opt.xyz', device)
    # box = torch.tensor([[vec.x / BOHR2NM, vec.y / BOHR2NM, vec.z / BOHR2NM] for vec in pdb.topology.getPeriodicBoxVectors()], device=device, requires_grad=True)

    prev = read('nvt.xyz')
    coords = torch.tensor(prev.get_positions() / BOHR2ANG, device=device, requires_grad=True)
    box = torch.tensor(prev.get_cell().array / BOHR2ANG, device=device, requires_grad=True)
    vel = prev.get_velocities()

    calc = CMMCalculator(system, top, coords, box)
    atoms = calc.atoms

    # Optimization
    # opt = LBFGS(atoms, logfile='opt.log')
    # opt.run(fmax=0.05, steps=2000)
    # atoms.write('opt.xyz')


    # NVT
    # temperature = 295.0
    # MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    
    atoms.set_velocities(vel)
    Stationary(atoms)

    # dyn = Langevin(
    #     atoms,
    #     timestep=1.0 * fs,
    #     temperature_K=temperature,
    #     friction=0.01 / fs,
    # )

    # traj = Trajectory('water216_nvt_295K.traj', 'a', atoms)
    # dyn.attach(traj.write, interval=50)
    
    # log = create_all_logger('water216_nvt_295K.log', atoms, dyn)
    # dyn.attach(log, interval=10)
    
    # dyn.run(2000)

    # atoms.write('nvt.xyz')


    dyn = NPTBerendsen(atoms, timestep=1.0 * fs, temperature_K=300.0,
                   taut=100 * fs, pressure_au=1.01325 * bar,
                   taup=1000 * fs, compressibility_au=4.57e-5 / bar)
    
    traj = Trajectory('water216_npt_300K_1atm.traj', 'a', atoms)
    dyn.attach(traj.write, interval=50)

    log = create_all_logger('water216_npt_300K_1atm.log', atoms, dyn)
    dyn.attach(log, interval=10)

    dyn.run(2000)
    atoms.write('npt.xyz')





