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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cmm.ase import CMMCalculator
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM, BOHR2ANG
torch.set_printoptions(precision=8)
torch.set_default_dtype(torch.float64)


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

         # record wall-clock elapsed time since logger start (HH:MM:SS.mmm)
        end = time.time()
        dur_ms = int((end - start) * 1000)
        hrs, rem = divmod(dur_ms, 3600 * 1000)
        mins, rem = divmod(rem, 60 * 1000)
        secs, ms = divmod(rem, 1000)
        timestr = f'{hrs:02}:{mins:02}:{secs:02}.{ms:03d}'

        # Write a line of data to the log file
        msg = f'{step:6d}  {time_fs:8.2f}  {temperature:10.2f}  {total_energy:12.6f}  {density:12.6f}  {timestr}'
        f.write(msg+'\n')
        print(msg)

    return log_all


if __name__ == '__main__':
    device = 'cuda'
    torch.set_default_dtype(torch.float64)

    _here = os.path.dirname(__file__)
    ff_path = os.environ.get('CMM_FF', os.path.join(_here, '..', 'tests', 'data', 'water_refit.xml'))
    pdb_path = os.environ.get('CMM_PDB', os.path.join(_here, '..', 'tests', 'data', 'water_216.pdb'))

    ff = ForceFieldXML(ff_path, device='cuda')
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)
    system = ff.parametrize(
        top, 
        use_fd_morse=True, use_polarization=True, polarization_tolerance=1e-7, ewald_tolerance=1e-9,
        use_hardness_change=False, use_lr_dispersion=True, cutoff_sr=9.0, use_switch=True, 
        use_customized_ops=True,
        use_pme=False
    )

    coords = torch.tensor(pdb.getPositions(asNumpy=True)._value / BOHR2NM, device=device, requires_grad=True)
    box = torch.tensor([[vec.x / BOHR2NM, vec.y / BOHR2NM, vec.z / BOHR2NM] for vec in pdb.topology.getPeriodicBoxVectors()], device=device, requires_grad=False)

    calc = CMMCalculator(system, top, coords, box, profile=False)
    atoms = calc.atoms

    # Optimization
    opt = LBFGS(atoms, logfile='opt.log')
    opt.run(fmax=0.05, steps=int(os.environ.get('CMM_OPT_STEPS', 2000)))
    atoms.write('opt.xyz')
    print("Optimization finished")

    calc.reset_profiler()

    # NVE
    calc.unset_minimization()
    temperature = 298.15
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    Stationary(atoms)

    dyn = VelocityVerlet(atoms, 1.0*fs)
    traj = Trajectory('water_md_nve_1ns.traj', 'a', atoms)
    dyn.attach(traj.write, interval=1000)
    log = create_all_logger('water216_1ns.log', atoms, dyn)
    dyn.attach(log, interval=100)

    dyn.run(int(os.environ.get('CMM_NVE_STEPS', 1000000)))

    calc.print_profiler()