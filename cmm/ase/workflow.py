import os, time
from typing import Optional

import torch

from ase import Atoms
from ase.io import Trajectory
from ase.optimize import LBFGS
from ase.md import Langevin
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import fs, bar, kB


from .LangevinMonteCarloBarostat import NPTLangevinMonteCarloBarostat
from .calculator import CMMCalculator


def create_dynamics_logger(filename: str, atoms: Atoms, dynamics: MolecularDynamics, verbose: bool = True):
    """
    Create a logging function for MD simulations that writes a header to the given file and
    returns a callable `log_all()` to record step, time (fs), temperature (K),
    total energy (eV), and density (g/cm3) at each interval.
    """
    # Open the log file and write the header line
    f = open(filename, 'w')
    header = '# step    time(fs)    temperature(K)    potential_energy(eV)  total_energy(eV)    density(g/cm3)   time'
    f.write(header+'\n')
    f.flush()
    if verbose:
        print(header)
    start = time.time()

    def log_all():
        # Get current step number
        step = dynamics.get_number_of_steps()
        # Get current simulation time in femtoseconds
        time_fs = dynamics.get_time() / fs

        # Compute temperature: T = 2 * E_kinetic / (3 * N * k_B)
        E_kin = atoms.get_kinetic_energy()
        temperature = atoms.get_temperature()

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
        msg = f'{step:6d}  {time_fs:8.2f}  {temperature:10.2f}  {E_pot:12.6f}  {total_energy:12.6f}  {density:12.6f}  {timestr}'
        f.write(msg+'\n')
        f.flush()
        if verbose:
            print(msg)

    return log_all


def run_simulation_workflow_with_atoms(
    atoms: Atoms,
    workdir: os.PathLike = '.',
    temperature: float = 298.15,
    pressure: float = 1.01325,
    verbose: bool = True,
    em_steps: int = 2000, em_fmax: float = 0.04,
    nvt_steps: int = 20000, nvt_time_step: float = 1.0, nvt_log_interval: int = 50, nvt_traj_interval: int = 100,
    npt_steps: int = 20000, npt_time_step: float = 1.0, npt_log_interval: int = 50, npt_traj_interval: int = 100,
    prod_steps: int = 20000, prod_time_step: float = 1.0, prod_log_interval: int = 50, prod_traj_interval: int = 100,
    prod_nvt: bool = False,
    gas_phase: bool = False,
):

    if not os.path.isdir(workdir):
        os.mkdir(workdir)

    # Optimization
    if em_steps > 0:
        opt = LBFGS(atoms, logfile=os.path.join(workdir, 'opt.log'))
        opt.run(fmax=em_fmax, steps=em_steps)
        atoms.write(os.path.join(workdir, 'opt.xyz'))
        if verbose:
            print("Optimization finished")
    
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    Stationary(atoms)
    fixcm = fixcm=False if gas_phase else True
    
    # NVT
    if nvt_steps > 0:
        dyn = Langevin(
            atoms, timestep=nvt_time_step * fs, 
            temperature_K=temperature, friction=0.005 / fs, fixcm=fixcm
        )

        traj = Trajectory(os.path.join(workdir, 'nvt.traj'), 'a', atoms)
        dyn.attach(traj.write, interval=nvt_traj_interval)
        
        log = create_dynamics_logger(os.path.join(workdir, 'nvt.log'), atoms, dyn, verbose)
        dyn.attach(log, interval=nvt_log_interval)
        dyn.run(nvt_steps)

        atoms.write(os.path.join(workdir, 'nvt.xyz'))
        if verbose:
            print("NVT finished")
    
    if npt_steps > 0:
        dyn = NPTLangevinMonteCarloBarostat(
            atoms, pressure_au=pressure * bar, timestep=npt_time_step * fs,
            temperature_K=temperature, friction=0.005 / fs, fixcm=fixcm
        )
    
        traj = Trajectory(os.path.join(workdir, 'npt.traj'), 'a', atoms)
        dyn.attach(traj.write, interval=npt_traj_interval)

        log = create_dynamics_logger(os.path.join(workdir, 'npt.log'), atoms, dyn, verbose)
        dyn.attach(log, interval=npt_log_interval)

        dyn.run(npt_steps)
        atoms.write(os.path.join(workdir, 'npt.xyz'))
        if verbose:
            print("NPT finished")
    
    if prod_steps > 0:
        if gas_phase or prod_nvt:
            dyn = Langevin(
                atoms, timestep=prod_time_step * fs, 
                temperature_K=temperature, friction=0.005 / fs, fixcm=fixcm
            )
        else:
            dyn = NPTLangevinMonteCarloBarostat(
                atoms, pressure_au=pressure * bar, timestep=prod_time_step * fs,
                temperature_K=temperature, friction=0.005 / fs, fixcm=fixcm
            )
        
        traj = Trajectory(os.path.join(workdir, 'prod.traj'), 'a', atoms)
        dyn.attach(traj.write, interval=prod_traj_interval)

        log = create_dynamics_logger(os.path.join(workdir, 'prod.log'), atoms, dyn, verbose)
        dyn.attach(log, interval=prod_log_interval)

        dyn.run(prod_steps)
        atoms.write(os.path.join(workdir, 'prod.xyz'))
        if verbose:
            print("Production finished")
    
    return atoms

    
def run_simulation_workflow(
    ff_path: os.PathLike,
    pdb_path: os.PathLike,
    workdir: os.PathLike = '.',
    temperature: float = 298.15,
    pressure: float = 1.01325,
    verbose: bool = True,
    em_steps: int = 2000, em_fmax: float = 0.04,
    nvt_steps: int = 20000, nvt_time_step: float = 1.0, nvt_log_interval: int = 50, nvt_traj_interval: int = 100,
    npt_steps: int = 20000, npt_time_step: float = 1.0, npt_log_interval: int = 50, npt_traj_interval: int = 100,
    prod_steps: int = 20000, prod_time_step: float = 1.0, prod_log_interval: int = 50, prod_traj_interval: int = 100,
    prod_nvt: bool = False,
    gas_phase: bool = False,
    device: Optional[str] = None,
    **kwargs
):
    import openmm.app as app
    from ..ffxml import ForceFieldXML
    from ..topology import Topology
    from ..units import BOHR2NM

    ff = ForceFieldXML(ff_path, device=device)
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)
    system = ff.parametrize(top, **kwargs)

    coords = torch.tensor(pdb.getPositions(asNumpy=True)._value / BOHR2NM, device=device, requires_grad=True)
    if not gas_phase:
        box = torch.tensor([[vec.x / BOHR2NM, vec.y / BOHR2NM, vec.z / BOHR2NM] for vec in pdb.topology.getPeriodicBoxVectors()], device=device, requires_grad=False)
    else:
        box = torch.tensor([
            [10.0 / BOHR2NM, 0.0 / BOHR2NM, 0.0 / BOHR2NM],
            [0.0 / BOHR2NM, 10.0 / BOHR2NM, 0.0 / BOHR2NM],
            [0.0 / BOHR2NM, 0.0 / BOHR2NM, 10.0 / BOHR2NM]
        ], device=device, requires_grad=False)

    calc = CMMCalculator(system, top, coords, box)
    atoms = calc.atoms

    return run_simulation_workflow_with_atoms(
        atoms, workdir, temperature, pressure, verbose,
        em_steps, em_fmax,
        nvt_steps, nvt_time_step, nvt_log_interval, nvt_traj_interval,
        npt_steps, npt_time_step, npt_log_interval, npt_traj_interval,
        prod_steps, prod_time_step, prod_log_interval, prod_traj_interval,
        prod_nvt, gas_phase
    )
    