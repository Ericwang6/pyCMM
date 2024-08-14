import pytest

import itertools
from pprint import pprint

import torch

from cmm.units import BOHR2NM, HARTREE2KJ, HARTREE2KCAL, BOHR2ANG
from cmm.multipole import computeLocal2GlobalRotationMatrix, rotateMultipoles, rotateQuadrupoles, computeCartesianQuadrupoles
from cmm.short_range import computeShortRangeEnergy
from cmm.dispersion import computeDispersion
from cmm.electrostatics import computePermElecAndPolarizationEnergy


@pytest.fixture
def water_data():
    coords = torch.tensor([
        [ 1.5165013870,  -0.0000008497,   0.1168590962],
        [ 0.5714469342,   0.0000007688,  -0.0477240756],
        [ 1.9206469769,   0.0000030303,  -0.7531309382],
        [-1.3965797657,   0.0000005579,  -0.1058991347],
        [-1.7503737705,  -0.7612400781,   0.3583839434],
        [-1.7503754020,   0.7612382608,   0.3583875091]
    ], dtype=torch.float64) / BOHR2ANG

    paramIndices = [0, 1, 1, 0, 1, 1]

    water_dimer_pairs = torch.tensor([[i, j] for i, j in itertools.product([0, 1, 2], [3, 4, 5])])
    water_dimer_pairs = torch.vstack((water_dimer_pairs, water_dimer_pairs[:, [1, 0]])).T

    # From: https://github.com/heindelj/CMM.jl/blob/main/src/components/parameters.jl#L45
    Z = torch.tensor([3.61565, 0.93619], dtype=torch.float64)
    mono = torch.tensor([-0.390896, 0.195448], dtype=torch.float64)
    qShell = mono - Z
    dipo = torch.tensor([
        [0.0,       0.0, -0.094298],
        [0.0910288, 0.0, -0.207851]
    ], dtype=torch.float64)
    
    quad_s = torch.tensor([
        # Q20,       Q21c,      Q21s, Q22c,       Q22s
        [-0.330685,  0.0,       0.0,  0.869923,   0.0],
        [-0.0739388, 0.0929482, 0.0,  0.00532425, 0.0]
    ], dtype=torch.float64)
    quad = computeCartesianQuadrupoles(quad_s)
    
    rotMatrix = computeLocal2GlobalRotationMatrix(
        coords, 
        coords[[1, 0, 0, 4, 3, 3]], 
        coords[[2, 2, 1, 5, 5, 4]], 
        coords[[-1, -1, -1, -1, -1, -1]], 
        torch.tensor([1, 0, 0, 1, 0, 0])
    )
    mPoles = rotateMultipoles(
        qShell[paramIndices],
        dipo[paramIndices],
        quad[paramIndices],
        rotMatrix
    ) * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3], dtype=torch.float64)

    b = torch.tensor([2.13358, 2.33322], dtype=torch.float64)

    # Pauli repulsion
    b_pauli = torch.tensor([2.1975, 1.96474], dtype=torch.float64)
    Kmono_pauli = torch.tensor([6.50923, 0.527804], dtype=torch.float64) / qShell
    Kdipo_pauli = torch.tensor([-5.61925, -0.515584], dtype=torch.float64)
    Kquad_pauli = torch.tensor([-1.56567, -0.440164], dtype=torch.float64)
    # Dispersion
    C6_disp = torch.tensor([35.8289, 1.98954], dtype=torch.float64)
    b_disp = torch.tensor([1.84302, 1.30993], dtype=torch.float64)
    # Polarization
    alpha = torch.tensor([
        [[4.45992, 0.0, 0.0], [0.0, 6.07259, 0.0], [0.0, 0.0, 4.55391]],
        [[2.22001, 0.0, 0.0], [0.0, 1.66835, 0.0], [0.0, 0.0, 0.183855]]
    ], dtype=torch.float64)[paramIndices]
    alpha = rotateQuadrupoles(alpha, rotMatrix)
    eta = torch.tensor([6.18699e-6, 0.561535], dtype=torch.float64) * 2
    groupCharges = torch.tensor([0.0, 0.0], dtype=torch.float64)
    # Exchange-polarization
    b_xpol = torch.tensor([2.73582, 2.04028], dtype=torch.float64)
    Kmono_xpol = torch.tensor([1.26592, 0.200089], dtype=torch.float64) / qShell
    Kdipo_xpol = torch.zeros((2,), dtype=torch.float64)
    Kquad_xpol = torch.zeros((2,), dtype=torch.float64)

    return (
        coords,
        water_dimer_pairs,
        Z[paramIndices],
        mPoles,
        b[paramIndices],
        b_pauli[paramIndices],
        Kmono_pauli[paramIndices],
        Kdipo_pauli[paramIndices],
        Kquad_pauli[paramIndices],
        C6_disp[paramIndices],
        b_disp[paramIndices],
        alpha,
        eta[paramIndices],
        groupCharges,
        b_xpol[paramIndices],
        Kmono_xpol[paramIndices],
        Kdipo_xpol[paramIndices],
        Kquad_xpol[paramIndices]
    )
    

def test_cmm(water_data):
    coords, pairs, Z, mPoles, b, b_pauli, Kmono_pauli, Kdipo_pauli, Kquad_pauli, C6_disp, b_disp, alpha, eta, groupCharges, b_xpol, Kmono_xpol, Kdipo_xpol, Kquad_xpol  = water_data

    drVec = coords[pairs[1]] - coords[pairs[0]]
    
    perm_elec, pol = computePermElecAndPolarizationEnergy(
        coords,
        [[0, 1, 2], [3, 4, 5]],
        mPoles,
        Z,
        b,
        True,
        alpha,
        eta,
        groupCharges
    )
    pol *= HARTREE2KCAL
    perm_elec *= HARTREE2KCAL

    pauli_pairwise = computeShortRangeEnergy(
        drVec,
        mPoles[pairs[0]], mPoles[pairs[1]],
        Kmono_pauli[pairs[0]], Kmono_pauli[pairs[1]],
        Kdipo_pauli[pairs[0]], Kdipo_pauli[pairs[1]],
        Kquad_pauli[pairs[0]], Kquad_pauli[pairs[1]],
        b_pauli[pairs[0]], b_pauli[pairs[1]]
    )
    pauli = torch.sum(pauli_pairwise) / 2 * HARTREE2KCAL

    disp_pairwise = computeDispersion(
        drVec, 
        C6_disp[pairs[0]], C6_disp[pairs[1]],
        b_disp[pairs[0]], b_disp[pairs[1]]
    )
    disp = torch.sum(disp_pairwise) / 2 * HARTREE2KCAL

    xpol_pairwise = computeShortRangeEnergy(
        drVec,
        mPoles[pairs[0]], mPoles[pairs[1]],
        Kmono_xpol[pairs[0]], Kmono_xpol[pairs[1]],
        Kdipo_xpol[pairs[0]], Kdipo_xpol[pairs[1]],
        Kquad_xpol[pairs[0]], Kquad_xpol[pairs[1]],
        b_xpol[pairs[0]], b_xpol[pairs[1]],
        False
    )
    xpol = torch.sum(xpol_pairwise) / 2 * HARTREE2KCAL
    
    xpol_ref = torch.tensor([-0.2740179677225849], dtype=torch.float64)
    assert torch.allclose(xpol, xpol_ref)

    pol_ref = torch.tensor([-0.7232598969865177], dtype=torch.float64)
    assert torch.allclose(pol, pol_ref)

    pauli_ref = torch.tensor([7.560565112146769], dtype=torch.float64)
    assert torch.allclose(pauli, pauli_ref)

    disp_ref = torch.tensor([-1.7327276064873118], dtype=torch.float64)
    assert torch.allclose(disp, disp_ref)

    perm_elec_ref = torch.tensor([-8.001893234755963], dtype=torch.float64)
    assert torch.allclose(perm_elec, perm_elec_ref, atol=0.005)