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

    def create_spcfw_system(box_vectors, use_lj_lr=True):
        system = mm.System()
        system.setDefaultPeriodicBoxVectors(*box_vectors)

        # Add particles for each atom in the system
        for i in range(modeller.topology.getNumAtoms()):
            system.addParticle(0)  # Mass will be set later

        # Set particle masses
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
        oxygen_sigma = 0.3165492 * unit.nanometer
        oxygen_epsilon = 0.1554253 * unit.kilocalorie_per_mole

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
        nonbonded_force.setNonbondedMethod(mm.NonbondedForce.PME)
        nonbonded_force.setEwaldErrorTolerance(1e-8)
        nonbonded_force.setUseDispersionCorrection(use_lj_lr)

        system.addForce(bond_force)
        system.addForce(angle_force)
        system.addForce(nonbonded_force)
        return system

    system = create_spcfw_system(pdb.topology.getPeriodicBoxVectors(), use_lj_lr=True)

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator)
    context.setPositions(modeller.positions)

    # Calculate energy
    state = context.getState(getEnergy=True)
    total_energy = state.getPotentialEnergy()

    # Get individual energy components by force group
    bond_energy = context.getState(getEnergy=True, groups={0}).getPotentialEnergy()
    angle_energy = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
    nonbonded_energy = context.getState(getEnergy=True, groups={2}).getPotentialEnergy()

    ### Now do the calculation with pyCMM ###
    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    system_file = os.path.join(os.path.dirname(__file__), "data/water_216.xyz")
    coords, atom_types, bonds, labels = read_from_tinker_xyz(system_file, requires_grad=True, device=device)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True, device=device)

    cm = CoordinateManager(coords, box, 9.0 / BOHR2ANG, labels=labels, max_neighbors=1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = SPCfw(ewald_tolerance=1e-10)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, {}, {}, ff.angle_params
    )

    energies = ff.evaluate(cm, topology, parameters)
    energies['total'].backward()
    #virial = torch.matmul(coords.grad.T, coords) + torch.matmul(box.grad.T, box)
    #print(torch.matmul(box.grad.T, box) * HARTREE2KJ)
    #stress = virial / (torch.det(cm.box) * BOHR2ANG**3)
    #print(virial * HARTREE2KJ)

    #print("perm_elec: ", energies['perm_elec'] * HARTREE2KCAL)
    #print("ewald: ", energies['ewald'] * HARTREE2KCAL)
    #print("lj: ", energies['lj'] * HARTREE2KCAL)
    #print("lj_lr_correction: ", energies['lj_lr_correction'] * HARTREE2KCAL)

    #print(f"Total Energy: {total_energy.value_in_unit(unit.kilocalorie_per_mole):.4f} kcal/mol / {energies['total'] * HARTREE2KCAL:.4f} kcal/mol")
    #print(f"Bond Energy: {bond_energy.value_in_unit(unit.kilocalorie_per_mole):.4f} kcal/mol / {energies['bond'] * HARTREE2KCAL:.4f} kcal/mol")
    #print(f"Angle Energy: {angle_energy.value_in_unit(unit.kilocalorie_per_mole):.4f} kcal/mol / {energies['angle'] * HARTREE2KCAL:.4f} kcal/mol")
    #print(f"Nonbonded Energy: {nonbonded_energy.value_in_unit(unit.kilocalorie_per_mole):.4f} kcal/mol / {(energies['total_elec'] + energies['lj']) * HARTREE2KCAL:.4f} kcal/mol")
    assert np.isclose(bond_energy.value_in_unit(unit.kilocalorie_per_mole), energies['bond'].clone().detach().cpu().numpy() * HARTREE2KCAL)
    assert np.isclose(angle_energy.value_in_unit(unit.kilocalorie_per_mole), energies['angle'].clone().detach().cpu().numpy() * HARTREE2KCAL)
    assert np.isclose(nonbonded_energy.value_in_unit(unit.kilocalorie_per_mole), (energies['total_elec'] + energies['lj']).clone().detach().cpu().numpy() * HARTREE2KCAL)