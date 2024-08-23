import pytest

import itertools
from pprint import pprint

import torch
from torch_scatter import scatter
import numpy as np

from cmm.units import BOHR2NM, HARTREE2KJ, HARTREE2KCAL, BOHR2ANG
from cmm.multipole import computeLocal2GlobalRotationMatrix, rotateMultipoles, rotateQuadrupoles, computeCartesianQuadrupoles
from cmm.short_range import computeShortRangeEnergy, scaleMultipoles, computePairwiseChargeTransfer
from cmm.dispersion import computeDispersion
from cmm.electrostatics import computePermElecAndPolarizationEnergy

import time


@pytest.fixture
def water_data():
    coords = torch.tensor(np.array([
        [ 1.5165013870,  -0.0000008497,   0.1168590962],
        [ 0.5714469342,   0.0000007688,  -0.0477240756],
        [ 1.9206469769,   0.0000030303,  -0.7531309382],
        [-1.3965797657,   0.0000005579,  -0.1058991347],
        [-1.7503737705,  -0.7612400781,   0.3583839434],
        [-1.7503754020,   0.7612382608,   0.3583875091]
    ]) / BOHR2ANG, dtype=torch.float32, requires_grad=True) 

    paramIndices = torch.LongTensor([0, 1, 1, 0, 1, 1])

    water_dimer_pairs = torch.tensor([[i, j] for i, j in itertools.product([0, 1, 2], [3, 4, 5])])
    water_dimer_pairs = torch.vstack((water_dimer_pairs, water_dimer_pairs[:, [1, 0]])).T

    # From: https://github.com/heindelj/CMM.jl/blob/main/src/components/parameters.jl#L45
    Z = torch.tensor([3.61565, 0.93619])
    mono = torch.tensor([-0.390896, 0.195448])
    qShell = mono - Z
    dipo = torch.tensor([
        [0.0,       0.0, -0.094298],
        [0.0910288, 0.0, -0.207851]
    ])
    
    quad_s = torch.tensor([
        # Q20,       Q21c,      Q21s, Q22c,       Q22s
        [-0.330685,  0.0,       0.0,  0.869923,   0.0],
        [-0.0739388, 0.0929482, 0.0,  0.00532425, 0.0]
    ])
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
    ) * torch.tensor([1, 1, 1, 1, 1/3, 2/3, 2/3, 1/3, 2/3, 1/3])

    b = torch.tensor([2.13358, 2.33322])
    param_elec = (Z[paramIndices], mPoles, b[paramIndices])

    # Pauli repulsion
    b_pauli = torch.tensor([2.1975, 1.96474])
    Kmono_pauli = torch.tensor([6.50923, 0.527804]) / qShell
    Kdipo_pauli = torch.tensor([-5.61925, -0.515584])
    Kquad_pauli = torch.tensor([-1.56567, -0.440164])
    param_pauli = (b_pauli[paramIndices], Kmono_pauli[paramIndices], Kdipo_pauli[paramIndices], Kquad_pauli[paramIndices])
    
    # Dispersion
    C6_disp = torch.tensor([35.8289, 1.98954])
    b_disp = torch.tensor([1.84302, 1.30993])
    param_disp = (C6_disp[paramIndices], b_disp[paramIndices])

    # Polarization
    alpha = torch.tensor([
        [[4.45992, 0.0, 0.0], [0.0, 6.07259, 0.0], [0.0, 0.0, 4.55391]],
        [[2.22001, 0.0, 0.0], [0.0, 1.66835, 0.0], [0.0, 0.0, 0.183855]]
    ])[paramIndices]
    alpha = rotateQuadrupoles(alpha, rotMatrix)
    eta = torch.tensor([6.18699e-6, 0.561535]) * 2
    groupCharges = torch.tensor([0.0, 0.0])
    param_pol = (alpha, eta[paramIndices], groupCharges)
    
    # Exchange-polarization
    b_xpol = torch.tensor([2.73582, 2.04028])
    Kmono_xpol = torch.tensor([1.26592, 0.200089]) / qShell
    Kdipo_xpol = torch.zeros((2,))
    Kquad_xpol = torch.zeros((2,))
    param_xpol = (b_xpol[paramIndices], Kmono_xpol[paramIndices], Kdipo_xpol[paramIndices], Kquad_xpol[paramIndices])

    # Charge Transfer
    b_ct = torch.tensor([1.89485, 2.36763])
    Kmono_ct_acc = torch.tensor([-0.67857, 1.36735]) / qShell
    Kdipo_ct_acc = torch.tensor([0.0, 0.0])
    Kquad_ct_acc = torch.tensor([0.0, 0.0])

    Kmono_ct_don = torch.tensor([0.757752, 0.00888982]) / qShell
    Kdipo_ct_don = torch.tensor([-0.512036, -0.0511668])
    Kquad_ct_don = torch.tensor([-0.208186, 0.0568152])

    eps_ct = torch.tensor([
        [1e15, 0.380979],
        [0.380979, 1e15]
    ])

    param_ct = (
        b_ct[paramIndices], 
        Kmono_ct_acc[paramIndices], Kdipo_ct_acc[paramIndices], Kquad_ct_acc[paramIndices], 
        Kmono_ct_don[paramIndices], Kdipo_ct_don[paramIndices], Kquad_ct_don[paramIndices], 
        eps_ct[paramIndices[water_dimer_pairs[0]], paramIndices[water_dimer_pairs[1]]]
    )

    return (
        coords,
        water_dimer_pairs,
        param_elec,
        param_pauli,
        param_disp,
        param_pol,
        param_xpol,
        param_ct,
    )
    

def test_cmm(water_data):
    coords, pairs, param_elec, param_pauli, param_disp, param_pol, param_xpol, param_ct = water_data

    drVec = coords[pairs[1]] - coords[pairs[0]]
    # Perm elec and polarization
    Z, mPoles, b = param_elec
    alpha, eta, groupCharges = param_pol
    # charge-transfer
    b_ct, Kmono_ct_acc, Kdipo_ct_acc, Kquad_ct_acc, Kmono_ct_don, Kdipo_ct_don, Kquad_ct_don, eps_ct = param_ct
    mPoles_ct_acc = scaleMultipoles(mPoles, Kmono_ct_acc, Kdipo_ct_acc, Kquad_ct_acc)
    mPoles_ct_don = scaleMultipoles(mPoles, Kmono_ct_don, Kdipo_ct_don, Kquad_ct_don)

    ct_direct_pairwise, dq_pairwise = computePairwiseChargeTransfer(
        drVec,
        mPoles_ct_acc[pairs[0]], mPoles_ct_acc[pairs[1]],
        mPoles_ct_don[pairs[0]], mPoles_ct_don[pairs[1]],
        b_ct[pairs[0]], b_ct[pairs[1]],
        eps_ct
    )
    ct_direct = torch.sum(ct_direct_pairwise) / 2 * HARTREE2KCAL
    dq = scatter(dq_pairwise, pairs[1])

    dq_groups = torch.tensor([torch.sum(dq[grp]) for grp in [[0, 1, 2], [3, 4, 5]]])

    ct_direct_ref = torch.tensor([-1.9459432858421248])
    assert torch.allclose(ct_direct, ct_direct_ref)

    perm_elec, pol, pol_ct = computePermElecAndPolarizationEnergy(
        coords,
        [[0, 1, 2], [3, 4, 5]],
        mPoles,
        Z,
        b,
        True,
        alpha,
        eta,
        groupCharges,
        groupCharges + dq_groups
    )
    pol *= HARTREE2KCAL
    perm_elec *= HARTREE2KCAL
    pol_ct *= HARTREE2KCAL

    perm_elec_ref = torch.tensor([-8.001893234755963])
    assert torch.allclose(perm_elec, perm_elec_ref, atol=0.001)
    pol_ref = torch.tensor([-0.7232598969865177])
    assert torch.allclose(pol, pol_ref)

    pol_ct_ref = torch.tensor([-0.4822699112017445])
    assert torch.allclose(pol_ct, pol_ct_ref, atol=0.001)

    # Pauli repulsion
    b_pauli, Kmono_pauli, Kdipo_pauli, Kquad_pauli = param_pauli
    mPoles_pauli = scaleMultipoles(mPoles, Kmono_pauli, Kdipo_pauli, Kquad_pauli)
    pauli_pairwise = computeShortRangeEnergy(
        drVec,
        mPoles_pauli[pairs[0]], mPoles_pauli[pairs[1]],
        b_pauli[pairs[0]], b_pauli[pairs[1]]
    )
    pauli = torch.sum(pauli_pairwise) / 2 * HARTREE2KCAL
    pauli_ref = torch.tensor([7.560565112146769])
    assert torch.allclose(pauli, pauli_ref)

    # dispersion
    C6_disp, b_disp = param_disp
    disp_pairwise = computeDispersion(
        drVec, 
        C6_disp[pairs[0]], C6_disp[pairs[1]],
        b_disp[pairs[0]], b_disp[pairs[1]]
    )
    disp = torch.sum(disp_pairwise) / 2 * HARTREE2KCAL
    disp_ref = torch.tensor([-1.7327276064873118])
    assert torch.allclose(disp, disp_ref)

    # exchange-polarization
    b_xpol, Kmono_xpol, Kdipo_xpol, Kquad_xpol = param_xpol
    mPoles_xpol = scaleMultipoles(mPoles, Kmono_xpol, Kdipo_xpol, Kquad_xpol)
    xpol_pairwise = computeShortRangeEnergy(
        drVec,
        mPoles_xpol[pairs[0]], mPoles_xpol[pairs[1]],
        b_xpol[pairs[0]], b_xpol[pairs[1]],
        False
    )
    xpol = torch.sum(xpol_pairwise) / 2 * HARTREE2KCAL
    xpol_ref = torch.tensor([-0.2740179677225849])
    assert torch.allclose(xpol, xpol_ref)
