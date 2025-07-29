import os
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import openmm.app as app
from cmm.forcefield import SystemNoCutoff, CMMForceField
from cmm.units import BOHR2ANG


def test_monomer_system():

    coords = torch.tensor(
        np.array([
            [-0.0000000000,     0.0000000000,     0.1167442842],
            [-0.7612369930,     0.0000000000,    -0.4669771368],
            [ 0.7612369930,     0.0000000000,    -0.4669771368],
        ]) / BOHR2ANG, 
        dtype=torch.float64, 
        requires_grad=True
    )

    
    dirname = os.path.dirname(__file__)
    top = app.PDBFile(os.path.join(dirname, 'data/water.pdb')).topology
    ff = CMMForceField(os.path.join(dirname, 'data/param_water.json'))
    system = ff.parametrize(top)

    print(system.evaluate_electric_properties(coords)['polarizability'])
