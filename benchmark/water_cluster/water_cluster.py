import os, sys
from pprint import pprint
os.environ["TORCH_COMPILE_DISABLE"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import pandas as pd
from tqdm import tqdm
import torch
import openmm.app as app
from ase.optimize import LBFGS

from cmm.ase import CMMCalculator
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM, BOHR2ANG, HARTREE2KCAL

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)


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


if __name__ == '__main__':
    device = 'cpu'
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'reference.csv'))

    ff_path = '/pscratch/sd/e/eric6/pycmm-dev/workspace_water/water_opt.xml'
    output_dir = 'output_new/'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ff = ForceFieldXML(ff_path, device=device)

    cutoff = 1000.0
    settings = {
        "use_fd_morse": True, "use_polarization": True, 'polarization_tolerance': 1e-7,
        'use_hardness_change': False, 'use_lr_dispersion': False,
        'cutoff_sr': cutoff, 'cutoff_lr': cutoff, 'use_switch': False,
        'use_customized_ops': False
    }

    rmsd = []
    energy_before_opt = []
    energy_after_opt = []

    monomer_top = Topology.fromPDB(os.path.join(os.path.dirname(__file__), 'water_monomer.pdb'), device=device)
    monomer_system = ff.parametrize(monomer_top, **settings)

    for struct in tqdm(df['structure']):
        box = torch.eye(3, device=device, requires_grad=False) * cutoff * 2.0 / BOHR2ANG
        coords = read_coords_from_xyz(os.path.join(os.path.dirname(__file__), f'xyz/{struct}.xyz'), device=device)
        top = Topology.fromPDB(os.path.join(os.path.dirname(__file__), f'pdb/{struct}.pdb'), device=device)
        system = ff.parametrize(top, **settings)
        num = coords.shape[0] // 3
        
        # calculate energy for each monomer before optimization
        before_opt = 0.0
        for i in range(num):
            monomer_system.last_induced_multipoles = torch.zeros(0, device=device)
            monomer_coords = coords.detach().clone()[i*3:i*3+3]
            before_opt += monomer_system.getEnergy(monomer_coords, box)['total'].item()

        before_opt = system.getEnergy(coords, box)['total'] - before_opt
        before_opt = before_opt.item()
        
        # optimization
        calc = CMMCalculator(system, top, coords, box, profile=False)
        atoms = calc.atoms

        opt = LBFGS(atoms, logfile=os.path.join(output_dir, f'{struct}_opt.log'))
        opt.run(fmax=0.001, steps=5000)
        atoms.write(os.path.join(output_dir, f'{struct}_opt.xyz'))

        # compute energy after optimization
        opt_coords = torch.tensor((atoms.get_positions()/ BOHR2ANG).tolist(), device=device, requires_grad=False)
        after_opt = 0.0
        for i in range(num):
            monomer_system.last_induced_multipoles = torch.zeros(0, device=device)
            monomer_coords = opt_coords.detach().clone()[i*3:i*3+3]
            after_opt += monomer_system.getEnergy(monomer_coords, box)['total']
        
        system.last_induced_multipoles = torch.zeros(0, device=device)
        after_opt = system.getEnergy(opt_coords, box)['total'] - after_opt
        after_opt = after_opt.item()

        
        rmsd.append(torch.sqrt(torch.sum((opt_coords - coords) ** 2) / coords.shape[0]).item() * BOHR2NM)
        energy_before_opt.append(before_opt * HARTREE2KCAL)
        energy_after_opt.append(after_opt * HARTREE2KCAL)
    
    df['CMM (before opt)'] = energy_before_opt
    df['CMM (after_opt)'] = energy_after_opt
    df['RMSD'] = rmsd
    df.to_csv(os.path.join(output_dir, 'result.csv'), index=None)
