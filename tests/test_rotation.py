import pytest
import os

import torch
torch.set_default_dtype(torch.float64)

import openmm as mm
import openmm.app as app

from cmm.units import BOHR2NM
from cmm.multipole import computeLocal2GlobalRotationMatrixBatch, rotateDipoles


def getReferenceData(pdb, xml):
    pdb = app.PDBFile(pdb)
    ff = app.ForceField(xml)
    system = ff.createSystem(pdb.topology)
    context = mm.Context(system, mm.LangevinIntegrator(298.0, 1.0, 2.0))
    context.setPositions(pdb.positions)
    force = [force for force in system.getForces() if isinstance(force, mm.AmoebaMultipoleForce)][0]
    
    mono, dipo, quad = [], [], []
    axisTypes = []
    zatoms, xatoms, yatoms = [], [], []
    for i in range(force.getNumMultipoles()):
        param = force.getMultipoleParameters(i)
        param[-1] = 0.0

        force.setMultipoleParameters(i, *param)

        mono.append(param[0]._value)
        dipo.append(param[1]._value)
        quad.append(param[2]._value)

        axisTypes.append(param[3])
        zatoms.append(param[4])
        xatoms.append(param[5])
        yatoms.append(param[6])
    
    dipoRef = torch.tensor([[d.x, d.y, d.z] for d in force.getLabFramePermanentDipoles(context)]) / BOHR2NM
    data = {
        "positions": torch.tensor(pdb.positions._value) / BOHR2NM,
        "axisTypes": torch.tensor(axisTypes, dtype=torch.long),
        'zAtoms': torch.tensor(zatoms, dtype=torch.long),
        'xAtoms': torch.tensor(xatoms, dtype=torch.long),
        'yAtoms': torch.tensor(yatoms, dtype=torch.long),
        "monoLoc": torch.tensor(mono),
        "dipoLoc": torch.tensor(dipo) / BOHR2NM,
        "quadLoc": torch.tensor(quad).reshape(-1, 3, 3) / (BOHR2NM * BOHR2NM) * 3,
        "dipoGlb": dipoRef
    }
    return data


@pytest.mark.parametrize('name', ['benzene', 'ethane', 'ammonia'])
def test_rotation(name: str):
    datadir = os.path.join(os.path.dirname(__file__), 'data')
    data = getReferenceData(
        os.path.join(datadir, f'{name}/{name}.pdb'),
        os.path.join(datadir, f'{name}/{name}.xml')
    )
    print('AxisTypes:', data['axisTypes'])
    rotMatrix = computeLocal2GlobalRotationMatrixBatch(
        data['positions'],
        data['zAtoms'],
        data['xAtoms'],
        data['yAtoms'],
        data['axisTypes']
    )
    dipoGlb = rotateDipoles(data['dipoLoc'], rotMatrix).squeeze(1)
    assert torch.allclose(dipoGlb, data['dipoGlb'])


