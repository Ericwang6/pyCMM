import torch
import os
import numpy as np
import time

from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.md import VelocityVerlet, MDLogger
from ase.md.nptberendsen import NPTBerendsen
from ase.md.langevin import Langevin
from ase.io import Trajectory, read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import fs, bar, kB

import openmm.app as app

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from cmm.interfaces import CMMCalculator
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM, BOHR2ANG


def read_coords_from_xyz(xyz, device):
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
    header = '# step    time(fs)    temperature(K)    total_energy(eV)    density(g/cm3)   time'
    f.write(header+'\n')
    print(header)
    start = time.time()

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

         # record time
        end = time.time()
        dur = int(end - start)
        hrs, secs = divmod(dur, 3600)
        mins, secs = divmod(secs, 60)
        timestr = f'{hrs:02}:{mins:02}:{secs:02}'

        # Write a line of data to the log file
        msg = f'{step:6d}  {time_fs:8.2f}  {temperature:10.2f}  {total_energy:12.6f}  {density:12.6f}  {timestr}'
        f.write(msg+'\n')
        print(msg)

    return log_all


if __name__ == '__main__':
    device = 'cuda'
    torch.set_default_dtype(torch.float64)

    ff_path = 'water_refit.xml'
    pdb_path = 'water_216.pdb'
    temperature = 298.15

    ff = ForceFieldXML(ff_path, device='cuda')
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)
    system = ff.parametrize(top, use_fd_morse=True, cutoff_sr=5.0, use_lr_dispersion=True, use_hardness_change=False)

    coords = torch.tensor(pdb.getPositions(asNumpy=True)._value / BOHR2NM, device=device, requires_grad=True)
    box = torch.tensor([[vec.x / BOHR2NM, vec.y / BOHR2NM, vec.z / BOHR2NM] for vec in pdb.topology.getPeriodicBoxVectors()], device=device, requires_grad=True)

    # prev = read('npt_298K_1000ps.traj', index=-1)
    # coords = torch.tensor(prev.get_positions() / BOHR2ANG, device=device, requires_grad=True)
    # box = torch.tensor(prev.get_cell().array / BOHR2ANG, device=device, requires_grad=True)
    # vel = prev.get_velocities()

    calc = CMMCalculator(system, top, coords, box)
    atoms = calc.atoms

    # Optimization
    opt = LBFGS(atoms, logfile='opt.log')
    opt.run(fmax=0.05, steps=2000)
    atoms.write('opt.xyz')
    print("Optimization finished")

    # NVT
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=1.0 * fs,
        temperature_K=temperature,
        friction=0.01 / fs,
    )

    traj = Trajectory('nvt_298K.traj', 'a', atoms)
    dyn.attach(traj.write, interval=50)
    
    log = create_all_logger('nvt_298K.log', atoms, dyn)
    dyn.attach(log, interval=10)
    
    dyn.run(2000)

    atoms.write('nvt.xyz')
    print("NVT finished")

    # NPT
    # atoms.set_velocities(vel)
    # Stationary(atoms)

    dyn = NPTBerendsen(atoms, timestep=1.0 * fs, temperature_K=temperature,
                   taut=100 * fs, pressure_au=1.01325 * bar,
                   taup=1000 * fs, compressibility_au=4.57e-5 / bar)
    
    traj = Trajectory('npt_298K_2000ps.traj', 'a', atoms)
    dyn.attach(traj.write, interval=1000)

    log = create_all_logger('npt_298K_2000ps.log', atoms, dyn)
    dyn.attach(log, interval=50)

    dyn.run(1000000)
    atoms.write('npt_298K_2000ps.xyz')