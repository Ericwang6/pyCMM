import pytest

import os
import itertools
from pprint import pprint

import torch
import openmm as mm
import openmm.app as app

from cmm.units import BOHR2NM, HARTREE2KJ
from cmm.multipole import computeLocal2GlobalRotationMatrix, rotateMultipoles, computePairwisePermElecEnergyNoDamp


def forcegroupify(system):
    forcegroups = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force.setForceGroup(i)
        forcegroups[force] = i
    return forcegroups


def getEnergyDecomposition(system, context):
    forcegroups = forcegroupify(system)
    energies = {}
    for f, i in forcegroups.items():
        energies[f.getName()] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
    return energies


def test_mpoles():
    water_dimer_pdb = os.path.join(os.path.dirname(__file__), 'water_dimer.pdb')
    ff = app.ForceField('amoeba2018.xml')
    pdb = app.PDBFile(water_dimer_pdb)
    system = ff.createSystem(pdb.topology)
    context = mm.Context(system, mm.LangevinIntegrator(300, 1.0, 1.0))
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
    
    force.updateParametersInContext(context)
    energies = getEnergyDecomposition(system, context)
    ene_ref = torch.tensor(energies['AmoebaMultipoleForce']._value, dtype=torch.float64)

    mono = torch.tensor(mono, dtype=torch.float64)
    dipo = torch.tensor(dipo, dtype=torch.float64) / BOHR2NM
    quad = torch.tensor(quad, dtype=torch.float64).reshape(-1, 3, 3) / (BOHR2NM * BOHR2NM) * 3

    axisTypes = torch.tensor(axisTypes, dtype=torch.long)
    zatoms, xatoms, yatoms = torch.tensor(zatoms), torch.tensor(xatoms), torch.tensor(yatoms)

    coords = torch.tensor(pdb.positions._value, dtype=torch.float64) / BOHR2NM
    rotMatrix = computeLocal2GlobalRotationMatrix(coords, coords[zatoms], coords[xatoms], coords[yatoms], axisTypes)
    mpoles = rotateMultipoles(mono, dipo, quad, rotMatrix)
    mpoles *= torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3], dtype=torch.float64)

    dipoRef = torch.tensor([[d.x, d.y, d.z] for d in force.getLabFramePermanentDipoles(context)], dtype=torch.float64) / BOHR2NM
    assert torch.allclose(mpoles[:, 1:4], dipoRef)

    # get pairs
    water_dimer_pairs = torch.tensor([[i, j] for i, j in itertools.product([0, 1, 2], [3, 4, 5])])
    water_dimer_pairs = torch.vstack((water_dimer_pairs, water_dimer_pairs[:, [1, 0]])).T
    
    drVec = coords[water_dimer_pairs[1]] - coords[water_dimer_pairs[0]]
    mPoles_i, mPoles_j = mpoles[water_dimer_pairs[0]], mpoles[water_dimer_pairs[1]]
    ene = torch.sum(computePairwisePermElecEnergyNoDamp(drVec, mPoles_i, mPoles_j)) / 2 * HARTREE2KJ
    assert torch.allclose(ene, ene_ref)
