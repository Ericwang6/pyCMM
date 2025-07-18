import os, sys
from tqdm import tqdm
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.units import BOHR2NM, BOHR2ANG

import openmm.app as app
from ase.io import read

if __name__ == '__main__':
    device = 'cuda'
    torch.set_default_dtype(torch.float64)

    ff_path = 'water.xml'
    pdb_path = '../water_216.pdb'
    # pdb_path = '../../tests/data/water_216.pdb'

    ff = ForceFieldXML(ff_path, device=device)
    pdb = app.PDBFile(pdb_path)
    top = Topology.fromOpenmm(pdb.topology, device)
    system = ff.parametrize(top)

    data = {'coords': [], 'box': [], 'charges': [], 'dipos': []}
    for end_time in [200, 400, 800, 1000]:
        # traj = read(f'nvt_298K_{end_time}ps_nonpt.traj', index=':')
        # if end_time < 1000:
        #     traj = traj[:-1]

        traj = read('water216_npt_298K_20ps.traj', index=':')
        for i, atoms in tqdm(enumerate(traj), total=len(traj)):
            coords = torch.tensor(atoms.get_positions() / BOHR2ANG, device=device, requires_grad=False)
            natoms = coords.shape[0]
            box = torch.tensor(atoms.get_cell().array / BOHR2ANG, device=device, requires_grad=False)
            system.getEnergy(coords, box)

            coords = coords.detach().cpu().numpy()
            box = box.detach().cpu().numpy()
            
            charges = system.last_perm_multipoles[:, 0].detach().cpu().numpy()
            dipos = system.last_perm_multipoles[:, 1:4].detach().cpu().numpy()
            ind_charges = system.last_induced_multipoles[:natoms].cpu().numpy()
            ind_dipos = system.last_induced_multipoles[natoms:4*natoms].view(-1, 3).detach().cpu().numpy()

            charges = charges + ind_charges
            dipos = dipos + ind_dipos

            data['coords'].append(coords)
            data['box'].append(box)
            data['charges'].append(charges)
            data['dipos'].append(dipos)
        break
    
    # for key in data:
    #     data[key] = np.array(data[key])
    data['time'] = np.linspace(0, 20, len(data['coords']))
    np.savez('water_npt_20ps_traj.npz', **data)