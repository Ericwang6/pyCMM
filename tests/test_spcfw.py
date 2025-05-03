import pytest

import torch
import numpy as np
import os

from cmm.units import BOHR2NM, HARTREE2KJ, HARTREE2KCAL, BOHR2ANG, DEBYE2EA
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.spcfw import SPCfw

from simtk.openmm import app
import simtk.openmm as mm
import simtk.unit as unit
from simtk.openmm.app import ForceField, PDBFile, Modeller

def test_spcfw_vs_openmm():
    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.pdb")
    pdb = PDBFile(system_file)
    modeller = Modeller(pdb.topology, pdb.positions)

    # Create a custom force field for SPC/fw water
    system = mm.System()

    box_vectors = pdb.topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*box_vectors)
    else:
        box_length = 1.8643 * unit.nanometer
        system.setDefaultPeriodicBoxVectors(
            mm.Vec3(box_length, 0, 0),
            mm.Vec3(0, box_length, 0),
            mm.Vec3(0, 0, box_length)
        )

    # Add particles for each atom in the system
    for i in range(modeller.topology.getNumAtoms()):
        system.addParticle(0)  # Mass will be set later

    # Set particle masses
    # SPC/fw uses standard atomic masses
    for i, atom in enumerate(modeller.topology.atoms()):
        if atom.element.symbol == 'O':
            system.setParticleMass(i, 15.99491461957 * unit.amu)
        elif atom.element.symbol == 'H':
            system.setParticleMass(i, 1.00782503223 * unit.amu)

    bond_force = mm.HarmonicBondForce()
    angle_force = mm.HarmonicAngleForce()
    nonbonded_force = mm.NonbondedForce()

    bond_force.setForceGroup(0)
    angle_force.setForceGroup(1)
    nonbonded_force.setForceGroup(2)

    # SPC/fw parameters
    bond_k = 1059.162 * unit.kilocalorie_per_mole/unit.angstrom**2
    bond_length = 1.012 * unit.angstrom
    angle_k = 75.90 * unit.kilocalorie_per_mole/unit.radian**2
    angle_value = 113.24 * unit.degree
    oxygen_charge = -0.82 * unit.elementary_charge
    hydrogen_charge = 0.41 * unit.elementary_charge
    oxygen_sigma = 0.0#0.3165492 * unit.nanometer
    oxygen_epsilon = 0.0#0.1554253 * unit.kilocalorie_per_mole

    # Keep track of bonds for nonbonded exclusions
    bonds_list = []

    # Add nonbonded parameters first
    for atom in modeller.topology.atoms():
        if atom.element.symbol == 'O':
            nonbonded_force.addParticle(oxygen_charge, oxygen_sigma, oxygen_epsilon)
        elif atom.element.symbol == 'H':
            nonbonded_force.addParticle(hydrogen_charge, 0.0, 0.0)

    # Add bond and angle interactions 
    for residue in modeller.topology.residues():
        if residue.name == 'HOH' or residue.name == 'WAT':
            oxygen = None
            hydrogens = []

            # Find O and H atoms in this water molecule
            for atom in residue.atoms():
                if atom.element.symbol == 'O':
                    oxygen = atom
                elif atom.element.symbol == 'H':
                    hydrogens.append(atom)

            if oxygen is not None and len(hydrogens) == 2:
                # Add O-H bonds
                for hydrogen in hydrogens:
                    bond_force.addBond(oxygen.index, hydrogen.index, bond_length, bond_k)
                    bonds_list.append((oxygen.index, hydrogen.index))

                # Add H-O-H angle
                angle_force.addAngle(hydrogens[0].index, oxygen.index, hydrogens[1].index, 
                                    angle_value, angle_k)

    nonbonded_force.createExceptionsFromBonds(bonds_list, 0.0, 0.0)

    # Add the forces to the system
    nonbonded_force.setCutoffDistance(0.9 * unit.nanometer)
    nonbonded_force.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
    #nonbonded_force.setEwaldErrorTolerance(1e-8)
    #nonbonded_force.setUseDispersionCorrection(True)
    
    system.addForce(bond_force)
    system.addForce(angle_force)
    system.addForce(nonbonded_force)

    # Now set up an integrator and simulation
    #integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
    #simulation = app.Simulation(modeller.topology, system, integrator)
    #simulation.context.setPositions(modeller.positions)

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator)
    context.setPositions(modeller.positions)
    #print(nonbonded_force.getPMEParametersInContext(context))

    # Calculate energy
    state = context.getState(getEnergy=True)
    total_energy = state.getPotentialEnergy()

    # Get individual energy components by force group
    bond_energy = context.getState(getEnergy=True, groups={0}).getPotentialEnergy()
    angle_energy = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
    nonbonded_energy = context.getState(getEnergy=True, groups={2}).getPotentialEnergy()

    # Print results
    print(f"Total Energy: {total_energy.value_in_unit(unit.kilocalorie_per_mole):.4f} kcal/mol")
    print(f"Bond Energy: {bond_energy.value_in_unit(unit.kilocalorie_per_mole):.4f} kcal/mol")
    print(f"Angle Energy: {angle_energy.value_in_unit(unit.kilocalorie_per_mole):.4f} kcal/mol")
    print(f"Nonbonded Energy: {nonbonded_energy.value_in_unit(unit.kilocalorie_per_mole):.4f} kcal/mol")

    # Minimize energy to remove bad contacts
    #simulation.minimizeEnergy(1e-2)

    # Run simulation
    #simulation.reporters.append(app.DCDReporter('trajectory.dcd', 1000))
    #simulation.reporters.append(app.StateDataReporter('output.dat', 1000, step=True, 
    #                                               potentialEnergy=True, temperature=True))
    #simulation.step(10000)  # Run for 10,000 steps

def test_spcfw():
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 200.0 / BOHR2ANG, requires_grad=True, device=device)

    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = SPCfw(ewald_tolerance=1e-15)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, {}, {}, ff.angle_params
    )

    energies = ff.evaluate(cm, topology, parameters)
    energies['total'].backward()
    for key in energies.keys():
        print(f"{key}: {float(energies[key] * HARTREE2KCAL):.4f}")
    print(energies['total'] * 627.51 / 216)