import pytest

import os
import itertools
from pprint import pprint

import torch
import openmm as mm
import openmm.app as app
from openmm.app import *
from openmm import *
from openmm.unit import nanometer, picosecond, picoseconds, kelvin
import numpy as np

from cmm.units import BOHR2NM, HARTREE2KJ, BOHR2ANG
from cmm.multipole import computeLocal2GlobalRotationMatrix, rotateMultipoles, computePairwisePermElecEnergyNoDamp, computeSphericalQuadrupoles, computeCartesianQuadrupoles
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM


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
    water_dimer_pdb = os.path.join(os.path.dirname(__file__), 'data/water_dimer_2.pdb')
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

def test_example():
    input_pdb = os.path.join(os.path.dirname(__file__), 'data/input.pdb')
    pdb = PDBFile(input_pdb)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()