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

def test_ewald():
    torch.set_default_dtype(torch.float64)
    water_216_pdb = os.path.join(os.path.dirname(__file__), 'data/water_216.pdb')
    pdb = app.PDBFile(water_216_pdb)
    ff = app.ForceField('amoeba2018.xml')
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.8 * nanometer,
        vdwCutoff=0.8 * nanometer,
        ewaldErrorTolerance=0.00001,
    )
    box_np = np.eye(3) * 18.643 / BOHR2ANG
    box = torch.tensor(box_np, dtype=torch.float64, requires_grad=True)
    context = mm.Context(system, mm.LangevinIntegrator(300, 1.0, 1.0))
    context.setPositions(pdb.positions)

    force = [force for force in system.getForces() if isinstance(force, mm.AmoebaMultipoleForce)][0]

    mono, dipo, quad = [], [], []
    axisTypes = []
    zatoms, xatoms, yatoms = [], [], []
    for i in range(force.getNumMultipoles()):
        param = force.getMultipoleParameters(i)
        param[-1] = 0.0 # Set polarization params to zero
        param[1]._value = np.zeros(3)
        param[2]._value = np.zeros(9)

        force.setMultipoleParameters(i, *param)

        mono.append(param[0]._value)
        dipo.append(param[1]._value)
        quad.append(param[2]._value)

        axisTypes.append(param[3])
        zatoms.append(param[4])
        xatoms.append(param[5])
        yatoms.append(param[6])
    
    #force.setPMEParameters(0.3, 29, 29, 29)
    force.updateParametersInContext(context)
    print(force.getPMEParametersInContext(context)[0] * BOHR2NM)
    print(force.getPMEParametersInContext(context))
    print(force.getCutoffDistance())
    energies = getEnergyDecomposition(system, context)
    ene_ref = torch.tensor(energies['AmoebaMultipoleForce']._value, dtype=torch.float64)
    print(ene_ref)

    mono = torch.tensor(np.array(mono), dtype=torch.float64)
    dipo = torch.tensor(np.array(dipo), dtype=torch.float64) / BOHR2NM
    quad = torch.tensor(np.array(quad), dtype=torch.float64).reshape(-1, 3, 3) / (BOHR2NM * BOHR2NM) * 3
    quad_s = computeSphericalQuadrupoles(quad)

    coords, atom_types, bonds = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216.xyz"), requires_grad=True)
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, dtype=torch.float64, requires_grad=True)
    #box = torch.tensor(np.eye(3) * 100.0 / BOHR2ANG, dtype=torch.float64, requires_grad=True)

    cm = CoordinateManager(coords, box, 8.0 / BOHR2ANG, 2048)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(cutoff_ewald=torch.tensor(8.0 / BOHR2ANG), use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    #energies = ff.evaluate(cm, topology, parameters)
    #print(energies['perm_elec'] * HARTREE2KJ)
    ff._raw_atomic_params['b_elec'][0] = 10000000000.0
    ff._raw_atomic_params['b_elec'][1] = 10000000000.0
    ff.mono[0] = mono[0]
    ff.mono[1] = mono[1]
    ff.dipo[0] = dipo[0]
    ff.dipo[1] = dipo[1]
    ff.quad_s[0] = quad_s[0]
    ff.quad_s[1] = quad_s[1]
    ff.rebuild_atomic_params()
    energies = ff.evaluate(cm, topology, parameters)
    # NOTE(JOE): Remember, the below is not exactly the undamped energy because of charge flux!
    #print(energies['perm_elec'] * HARTREE2KJ)

    # TODO: Apply PBC to the distance vectors.
    # Also apply proper damping to the interaction.
    # Just use the current force field implementation and
    # save the components of the interaction that we need to compare to
    # other interactions. Gives a good opportunity to add the ability
    # to update parameters and then propagate that to the parameterizer.
    #water_dimer_pairs = torch.tensor([[i, j] for i, j in itertools.product([0, 1, 2], [3, 4, 5])])
    #water_dimer_pairs = torch.vstack((water_dimer_pairs, water_dimer_pairs[:, [1, 0]])).T
    #
    #drVec = coords[water_dimer_pairs[1]] - coords[water_dimer_pairs[0]]
    #mPoles_i, mPoles_j = mpoles[water_dimer_pairs[0]], mpoles[water_dimer_pairs[1]]
    #ene = torch.sum(computePairwisePermElecEnergyNoDamp(drVec, mPoles_i, mPoles_j)) / 2 * HARTREE2KJ
    #assert torch.allclose(ene, ene_ref)