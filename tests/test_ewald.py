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

from openmm.openmm import System, VerletIntegrator, NonbondedForce, Context, State, AmoebaMultipoleForce
from openmm.unit import nanometer, picosecond, picoseconds, kelvin
from openmm.unit import AVOGADRO_CONSTANT_NA
import openmm.app as app

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
    # NOTE(JOE): I set these Z values because the shell charge is Q = q - Z.
    # Ordinarily, then, by setting Z=0 the total charge of a fragment will be
    # zero. In this case, by setting the core charge for Cl- to -1.0, we get
    # a zero shell charge. Same for Na+ with core charge of +1.0. This then tests
    # just the long-range electrostatics.
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
