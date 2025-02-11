import pytest
import torch
import numpy as np
import os
import time
import torch._dynamo as dynamo

from cmm.units import HARTREE2KCAL, BOHR2ANG, BOHR2NM, HARTREE2KJ
from cmm.misc_utils import read_from_tinker_xyz
from cmm.cmm_water import CMMWater
from cmm.coordinate_manager import CoordinateManager
from cmm.topology import Topology
from cmm.parameters import Parameterizer
from cmm.force_field import CMM

from openmm.openmm import System, VerletIntegrator, NonbondedForce, Context, State
from openmm.unit import AVOGADRO_CONSTANT_NA

# Ported from https://github.com/openmm/openmm/blob/39808f12cb7dc0465748cc3884b50594ef568523/tests/TestEwald.h
def test_ewald_exact():
    torch.set_default_dtype(torch.float64)
    # Use a NaCl crystal to compare the calculated and Madelung energies
    numParticles = 1000
    cutoff = 1.0
    boxSize = 2.82
    ewaldTol = 1e-5

    #system = System()
    #for _ in range(numParticles // 2):
    #    system.addParticle(22.99)
    #for _ in range(numParticles // 2):
    #    system.addParticle(35.45)
    #
    #integrator = VerletIntegrator(0.01)
    #nonbonded = NonbondedForce()
    #for _ in range(numParticles // 2):
    #    nonbonded.addParticle(1.0, 0.0, 0.0)
    #for _ in range(numParticles // 2):
    #    nonbonded.addParticle(-1.0, 0.0, 0.0)
    #nonbonded.setNonbondedMethod(NonbondedForce.Ewald)
    #nonbonded.setCutoffDistance(cutoff)
    #system.setDefaultPeriodicBoxVectors(np.array([boxSize, 0, 0]), np.array([0, boxSize, 0]), np.array([0, 0, boxSize]))
    #nonbonded.setEwaldErrorTolerance(ewaldTol)
    #system.addForce(nonbonded)
    #context = Context(system, integrator)

    # Loaded positions are in nm
    positions = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/nacl_crystal.txt"), dtype=np.float64)
    #context.setPositions(positions)

    #state = context.getState(energy=True, forces=True)

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
    #print("OpenMM Ewald: ", state.getPotentialEnergy())
    #print("Exact: ", exactEnergy._value)

    coords = torch.tensor(positions / BOHR2NM, dtype=torch.float64, requires_grad=True)
    bonds = np.array([], dtype=np.float64)
    atom_type_names = ["" for i in range(coords.size(0))]
    for i in range(coords.size(0) // 2):
        atom_type_names[i] = "Na+"
    for i in range(coords.size(0) // 2, coords.size(0)):
        atom_type_names[i] = "Cl-"

    box = torch.tensor(np.eye(3) * 28.2 / BOHR2ANG, dtype=torch.float64, requires_grad=True)

    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 2048)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs()
    ff = CMM(cutoff_ewald=torch.tensor(10.0 / BOHR2ANG), ewald_tolerance=torch.tensor(1e-10), use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )

    energies = ff.evaluate(cm, topology, parameters)

    assert torch.isclose(torch.tensor(exactEnergy._value), energies["ewald"] * HARTREE2KJ)

def test_direct_ewald():
    torch.set_default_dtype(torch.float64)

    coords_no_grad, _, _ = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=False)
    coords, atom_types, bonds = read_from_tinker_xyz(os.path.join(os.path.dirname(__file__), "data/water_dimer.xyz"), requires_grad=True)
    atom_indices_to_names = {0: "O_water", 1: "H_water"}
    atom_type_names = [atom_indices_to_names[int(atom_types[i])] for i in range(len(atom_types))]
    box = torch.tensor(np.eye(3) * 100, dtype=torch.float64, requires_grad=True)
    
    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    ff = CMM(use_ewald=True)
    parameters = Parameterizer(
        atom_type_names, pairs, topology.angle_atoms,
        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    )
    energies_ff = ff.evaluate(cm, topology, parameters)
    #total_ref = torch.tensor([-4.768231511534177 / HARTREE2KCAL])
    #assert torch.allclose(energies_ff['tot'], total_ref)

    #def get_total_energy(coords: torch.Tensor):
    #    cm = CoordinateManager(coords, box, 10.0 / BOHR2ANG, 1024)
    #    topology = Topology(bonds, cm.neighbor_list, coords.size(0))
    #    pairs, _, _ = cm.get_distances_vectors_and_pairs()
    #    ff = CMM()
    #    parameters = Parameterizer(
    #        atom_type_names, pairs, topology.angle_atoms,
    #        ff.atomic_params, ff.pair_params, ff.pair_pair_params, ff.pair_angle_params, ff.angle_params
    #    )
    #    energies_ff = ff.evaluate(cm, topology, parameters)
    #    total_energy = energies_ff['tot']
    #    return total_energy
    #
    #grads_fd_1 = finite_difference(coords_no_grad, get_total_energy, h=1e-5)
    #energies_ff['tot'].backward(retain_graph=True)
    #grads_ad_1 = coords.grad.clone()
    #if not torch.allclose(grads_ad_1, grads_fd_1):
    #    print(grads_ad_1 - grads_fd_1)
    #
    #assert torch.allclose(grads_ad_1, grads_fd_1)