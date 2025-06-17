import pytest
import torch
import numpy as np
import os
import time
import torch._dynamo as dynamo

from cmm.units import HARTREE2KCAL, BOHR2ANG, BOHR2NM, HARTREE2KJ
from cmm.misc_utils import read_from_tinker_xyz
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM
from cmm.multipole import computeSphericalQuadrupoles

import openmm as mm
import openmm.app as app
from openmm.unit import AVOGADRO_CONSTANT_NA

# Ported from https://github.com/openmm/openmm/blob/39808f12cb7dc0465748cc3884b50594ef568523/tests/TestEwald.h
def test_ewald_exact():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Use a NaCl crystal to compare the calculated and Madelung energies
    numParticles = 1000
    boxSize = 28.2

    # Loaded positions are in nm
    positions = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/nacl_crystal.txt"), dtype=np.float64)

    #   The potential energy of an ion in a crystal is 
    #   E = - (M*e^2/ 4*pi*epsilon0*a0),
    #   where 
    #   M            :    Madelung constant (dimensionless, for FCC cells such as NaCl it is 1.7476)
    #   e            :    1.6022 × 10−19 C
    #   4*pi*epsilon0:     1.112 × 10−10 C²/(J m)
    #   a0           :    0.282 x 10-9 m (perfect cell)
    # 
    #   E is then the energy per pair of ions, so for our case
    #   E has to be divided by 2 (per ion), multiplied by N(avogadro), multiplied by number of particles, and divided by 1000 for kJ
    
    # Madelung constant for NaCl can be computed by series:
    # See https://en.wikipedia.org/wiki/Madelung_constant
    M = 0.0
    m_max = 13
    n_max = 13
    for m in range(1, m_max, 2):
        for n in range(1, n_max, 2):
            M += 1 / np.cosh(0.5 * np.pi * np.sqrt(m*m + n*n))**2 # 1 / cosh === sech
    M *= 12 * np.pi
    
    FOUR_PI_EPS = 1.1126500562e-10
    CELL_LENGTH = 0.282e-9
    COULOMB = 1.602176634e-19
    exactEnergy = -(M * COULOMB * COULOMB  * AVOGADRO_CONSTANT_NA * numParticles) / (FOUR_PI_EPS * CELL_LENGTH * 2 * 1000)

    coords = torch.tensor(positions / BOHR2NM, requires_grad=True)
    bonds = np.empty((2, 0), dtype=np.float64)
    atom_type_names = ["" for i in range(coords.size(0))]
    labels = ["" for i in range(coords.size(0))]
    for i in range(coords.size(0) // 2):
        atom_type_names[i] = "Na+"
        labels[i] = "Na"
    for i in range(coords.size(0) // 2, coords.size(0)):
        atom_type_names[i] = "Cl-"
        labels[i] = "Cl"

    box = torch.tensor(np.eye(3) * boxSize / BOHR2ANG, requires_grad=True)

    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(ewald_tolerance=torch.tensor(1e-15), use_ewald=True, use_polarization=False)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    ff.alpha[3] = torch.diag(torch.tensor([0.0000000001 for _ in range(3)]))
    ff.alpha[7] = torch.diag(torch.tensor([0.0000000001 for _ in range(3)]))
    ff._raw_atomic_params['b_elec'][3] = torch.tensor([10000000000.0])
    ff._raw_atomic_params['b_elec'][7] = torch.tensor([10000000000.0])
    ff.pair_params[("Na+", "Cl-")]['b_elec'] = torch.tensor([10000000000.0])
    ff._raw_atomic_params['Z'][3] = -1.0
    ff._raw_atomic_params['Z'][7] = 1.0
    # NOTE(JOE): ^^^ I set these Z values because the shell charge is Q = q - Z.
    # In this case, by setting the core charge for Cl- to -1.0, we get
    # a zero shell charge. Same for Na+ with core charge of +1.0. Combined with
    # a very large b-value, charge penetration goes to zero, and this just tests
    # the long-range electrostatics.
    ff.rebuild_atomic_params()

    energies = ff.evaluate(cm, topology, parameters)
    elec_energy_cmm = (energies["perm_elec"] + energies["ewald"]) * HARTREE2KJ
    assert torch.isclose(torch.tensor(exactEnergy._value), elec_energy_cmm)

def test_multipolar_ewald_water_mchem_reference():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    coords, atom_types, bonds, labels = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_216_mchem.xyz"), requires_grad=True)
    
    # Normally, the parser should enforce just returning the names of atom types
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 18.643 / BOHR2ANG, requires_grad=True)
    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 7.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True, use_polarization=False, cutoff_ewald=torch.tensor(9.0 / BOHR2ANG), ewald_tolerance=torch.tensor(1e-15))
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    ff._raw_atomic_params['b_elec'][0] = 10000000000.0
    ff._raw_atomic_params['b_elec'][1] = 10000000000.0
    ff.mono[0] = -0.51966
    ff.mono[1] = 0.25983
    ff.dipo[0] = torch.tensor([0.0, 0.0, 0.14279])
    ff.dipo[1] = torch.tensor([-0.03859, 0.0, -0.05818])
    quad_O = torch.tensor([[0.56803, 0.0, 0.0], [0.0, -0.65906, 0.0], [0.0, 0.0, 0.09103]])
    quad_H = torch.tensor([[-0.01730, 0.0, 0.00007], [0.0, -0.07631, 0.0], [0.00007, 0.0, 0.09361]])
    ff.quad_s[0] = computeSphericalQuadrupoles(quad_O.unsqueeze(0))
    ff.quad_s[1] = computeSphericalQuadrupoles(quad_H.unsqueeze(0))
    ff.pair_params[("O_water", "H_water")]["j_cf"] = torch.tensor([0.0])
    ff.pair_pair_params[(("O_water", "H_water"), ("O_water", "H_water"),)]["j_cf_bb"] = torch.tensor([0.0])
    ff.angle_params[("H_water", "O_water", "H_water")]["j_cf_angle"] = torch.tensor([0.0])
    ff.rebuild_atomic_params()
    
    energies = ff.evaluate(cm, topology, parameters)

    # Reference values from mchem
    # perm_elec is the actual real space plus the negative of the
    # masked interactions which are needed to cancel out their inclusion in reciprocal space.
    # Ewald is the sum of self interactions and reciprocal space.
    ene_elec = energies["perm_elec"] * HARTREE2KCAL
    ene_ewald = energies["ewald"] * HARTREE2KCAL
    assert torch.isclose(ene_elec + ene_ewald, torch.tensor(-2416.42428445285), 1e-3)

def test_long_range_polarization():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coords_file = os.path.join(os.path.dirname(__file__), "data/nacl_amorph.txt")
    
    box_size = 28.2
    box_vectors = [
        mm.Vec3(box_size, 0, 0),
        mm.Vec3(0, box_size, 0), 
        mm.Vec3(0, 0, box_size)
    ] * mm.unit.angstrom
    system, topology, positions = create_nacl_system(coords_file, box_vectors)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_size, 0, 0) * mm.unit.angstrom,
        mm.Vec3(0, box_size, 0) * mm.unit.angstrom,
        mm.Vec3(0, 0, box_size) * mm.unit.angstrom
    )
    print(f"Created system with {system.getNumParticles()} particles")
    
    simulation, potential_energy = calculate_single_point_energy(system, topology, positions)
    print(f"Single point potential energy: {potential_energy}")

    force = system.getForces()[0]
    dips = force.getInducedDipoles(simulation.context)
    #print(dips)

    # Setup CMM calculation #
    positions = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/nacl_amorph.txt"), dtype=np.float64)
    numParticles = len(positions)
    coords = torch.tensor(positions / BOHR2NM, requires_grad=True)
    bonds = np.empty((2, 0), dtype=np.float64)
    atom_type_names = ["" for i in range(coords.size(0))]
    labels = ["" for i in range(coords.size(0))]
    for i in range(coords.size(0) // 2):
        atom_type_names[i] = "Na+"
        labels[i] = "Na"
    for i in range(coords.size(0) // 2, coords.size(0)):
        atom_type_names[i] = "Cl-"
        labels[i] = "Cl"

    box = torch.tensor(np.eye(3) * box_size / BOHR2ANG, requires_grad=True)

    topology = Topology(bonds, coords.size(0), device)
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, labels, topology.all_intramolecular_pairs)
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(ewald_tolerance=torch.tensor(1e-8), use_ewald=True, use_polarization=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    ff.alpha[3] = torch.diag(torch.tensor([35.0 for _ in range(3)]))
    ff.alpha[7] = torch.diag(torch.tensor([1.0 for _ in range(3)]))
    ff._raw_atomic_params['b_elec'][3] = torch.tensor([10000000000.0])
    ff._raw_atomic_params['b_elec'][7] = torch.tensor([10000000000.0])
    ff.pair_params[("Na+", "Cl-")]['b_elec'] = torch.tensor([10000000000.0])
    ff._raw_atomic_params['Z'][3] = -1.0
    ff._raw_atomic_params['Z'][7] = 1.0
    ff._raw_atomic_params["alpha_damp_exponent"][3] = 0.0
    ff._raw_atomic_params["alpha_damp_exponent"][7] = 0.0
    ff._raw_atomic_params["alpha_damp_max"][3] = 0.0
    ff._raw_atomic_params["alpha_damp_max"][7] = 0.0
    # NOTE(JOE): ^^^ I set these Z values because the shell charge is Q = q - Z.
    # In this case, by setting the core charge for Cl- to -1.0, we get
    # a zero shell charge. Same for Na+ with core charge of +1.0. Combined with
    # a very large b-value, charge penetration goes to zero, and this just tests
    # the long-range electrostatics.
    ff.rebuild_atomic_params()

    energies = ff.evaluate(cm, topology, parameters)
    elec_energy_cmm = (energies["perm_elec"] + energies["ewald"]) * HARTREE2KJ
    print(elec_energy_cmm)
    print(ff.last_induced_multipoles)
    # HERE: Figure out why the energies DO AGREE when only using permanent electrostatics
    # but do not agree when including polarization. After that, check on the value of the virial
    # as computed by OpenMM compared to that computed here.

def calculate_single_point_energy(system, topology, positions):
    """
    Calculate single point energy for the given configuration.
    """
    integrator = mm.VerletIntegrator(1.0*mm.unit.femtosecond)
    
    # Create simulation
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    
    # Get energy without any minimization or dynamics
    state = simulation.context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    
    return simulation, potential_energy

def create_nacl_system(coordinates_file, box_vectors):
    topology = app.Topology()
    chain = topology.addChain()
    coords_data = np.loadtxt(coordinates_file) # (coordinates in nanometers)
    n_atoms = len(coords_data)
    positions = []
    atoms = []
    
    for i, coord in enumerate(coords_data):
        x, y, z = coord[0], coord[1], coord[2]
        element = 'Na' if i < (n_atoms // 2) else 'Cl'
        
        # Add atom to topology
        residue = topology.addResidue(f"ION{i+1}", chain)
        atom = topology.addAtom(element, app.Element.getBySymbol(element), residue)
        atoms.append(atom)
        
        # Store position (coordinates already in nanometers)
        positions.append(mm.Vec3(x, y, z) * mm.unit.nanometer)
    
    topology.setPeriodicBoxVectors(box_vectors)
    
    # Create system
    system = mm.System()
    
    # Add particles
    for i, atom in enumerate(atoms):
        if atom.element.symbol == 'Na':
            mass = 22.99 * mm.unit.amu
        elif atom.element.symbol == 'Cl':
            mass = 35.45 * mm.unit.amu
        system.addParticle(mass)
    
    # Add AMOEBA multipole force for electrostatics and polarization
    amoeba_force = mm.AmoebaMultipoleForce()
    
    # Set nonbonded method for periodic systems
    amoeba_force.setNonbondedMethod(mm.AmoebaMultipoleForce.PME)
    amoeba_force.setCutoffDistance(1.2 * mm.unit.nanometer)
    amoeba_force.setEwaldErrorTolerance(1e-7)
    
    amoeba_force.setMutualInducedMaxIterations(500)
    amoeba_force.setMutualInducedTargetEpsilon(1e-6)
    
    for i, atom in enumerate(atoms):
        if atom.element.symbol == 'Na':
            charge = +1.0
            polarizability = 1.0 * mm.unit.bohr**3
            thole = 1000000000000.0
        elif atom.element.symbol == 'Cl':
            charge = -1.0
            polarizability = 35.0 * mm.unit.bohr**3
            thole = 1000000000000.0
        
        amoeba_force.addMultipole(
            charge * mm.unit.elementary_charge,  # charge
            np.zeros(3),  # dipole moments (x,y,z)
            np.zeros(6),   # quadrupole moments
            mm.AmoebaMultipoleForce.NoAxisType,
            -1, -1, -1,
            thole,
            thole,
            polarizability
        )
    
    system.addForce(amoeba_force)
    
    return system, topology, positions