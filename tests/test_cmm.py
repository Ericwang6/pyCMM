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
from cmm.cmm_water import CMMWater

def finite_difference(coords: torch.Tensor, f, h: float = 1e-5):
    grads_fd = torch.zeros(coords.shape)
    for i in range(coords.shape[0]):
        for w in range(coords.shape[1]):
            coords[i, w] += h
            f_plus_h = f(coords)

            coords[i, w] -= 2 * h
            f_minus_h = f(coords)
            coords[i, w] += h

            grads_fd[i, w] = (f_plus_h - f_minus_h) / (2 * h)
    return grads_fd

def get_water_dimer_coords(requires_grad=True):
    coords = torch.tensor(np.array([
        [0.0031771858, 1.4710501499, -0.0034222052],
        [0.0981342864, 0.5090994249, -0.0041139499],
        [0.8976520298, 1.8147098331, 0.0031728568],
        [-0.0036002768, -1.3547622039, 0.0027150961],
        [-0.492886242, -1.6733692175, 0.7647713563],
        [-0.4948459831, -1.6611969865, -0.763123154],
    ]) / BOHR2ANG, dtype=torch.float64, requires_grad=requires_grad)
    return coords

def water_data(coords: torch.Tensor):
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
    Kmono_pauli = torch.tensor([6.50923, 0.527804])
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
    Kmono_xpol = torch.tensor([1.26592, 0.200089])
    Kdipo_xpol = torch.zeros((2,))
    Kquad_xpol = torch.zeros((2,))
    param_xpol = (b_xpol[paramIndices], Kmono_xpol[paramIndices], Kdipo_xpol[paramIndices], Kquad_xpol[paramIndices])

    # Charge Transfer
    b_ct = torch.tensor([1.89485, 2.36763])
    Kmono_ct_acc = torch.tensor([-0.67857, 1.36735])
    Kdipo_ct_acc = torch.tensor([0.0, 0.0])
    Kquad_ct_acc = torch.tensor([0.0, 0.0])

    Kmono_ct_don = torch.tensor([0.757752, 0.00888982])
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
        water_dimer_pairs,
        param_elec,
        param_pauli,
        param_disp,
        param_pol,
        param_xpol,
        param_ct,
    )

def test_nonbonded_interactions():
    torch.set_default_dtype(torch.float64)

    coords = get_water_dimer_coords(requires_grad=False)

    model = CMMWater(2, do_polarization=True)
    box = torch.tensor(np.eye(3) * 100, dtype=torch.float64, requires_grad=True)
    energies = model.computeEnergy(coords, box)

    ct_direct = energies['ct_direct'] * HARTREE2KCAL
    perm_elec = energies['perm_elec'] * HARTREE2KCAL
    pol_ct = energies['pol'] * HARTREE2KCAL
    pauli = energies['pauli'] * HARTREE2KCAL
    disp = energies['disp'] * HARTREE2KCAL
    xpol = energies['xpol'] * HARTREE2KCAL

    ct_direct_ref = torch.tensor([-2.777994825946152])
    perm_elec_ref = torch.tensor([-9.705778546689396])
    pol_ct_ref = torch.tensor([-0.56935496823574])
    pauli_ref = torch.tensor([10.70659662118688])
    disp_ref = torch.tensor([-2.054389376337415])
    xpol_ref = torch.tensor([-0.4036285834810728])
    assert torch.allclose(ct_direct, ct_direct_ref)
    assert torch.allclose(perm_elec, perm_elec_ref)
    assert torch.allclose(pol_ct, pol_ct_ref)
    assert torch.allclose(pauli, pauli_ref)
    assert torch.allclose(disp, disp_ref)
    assert torch.allclose(xpol, xpol_ref)

def test_bonded_params_with_fd_morse():
    torch.set_default_dtype(torch.float64)

    coords = get_water_dimer_coords(requires_grad=False)

    model = CMMWater(2, do_polarization=True)
    box = torch.tensor(np.eye(3) * 100, dtype=torch.float64, requires_grad=True)
    energies = model.computeEnergy(coords, box)
    print(energies)

    #ct_direct_ref = torch.tensor([-1.9459432858421248])
    #assert torch.allclose(ct_direct, ct_direct_ref)

def test_electrostatic_and_pol_gradients():
    torch.set_default_dtype(torch.float64)

    coords = get_water_dimer_coords()

    def get_elec_and_pol_energy(coords: torch.Tensor):
        pairs, param_elec, param_pauli, param_disp, param_pol, param_xpol, param_ct = water_data(coords)

        # Perm elec and polarization parameters
        Z, mPoles, b = param_elec
        alpha, eta, groupCharges = param_pol
        perm_elec, pol = computePermElecAndPolarizationEnergy(
            coords,
            [[0, 1, 2], [3, 4, 5]],
            mPoles,
            Z,
            b,
            True,
            alpha,
            eta,
            groupCharges,
        )
        energy = perm_elec + pol
        return energy
    
    energy = get_elec_and_pol_energy(coords)
    energy.backward()
    grads_ad = coords.grad

    coords = get_water_dimer_coords(requires_grad=False)
    grads_fd = finite_difference(coords, get_elec_and_pol_energy, h=1e-5)
    assert torch.allclose(grads_ad, grads_fd)

def test_dispersion_gradients():
    torch.set_default_dtype(torch.float64)

    coords = get_water_dimer_coords()

    def get_dispersion_energy(coords: torch.Tensor):
        pairs, param_elec, param_pauli, param_disp, param_pol, param_xpol, param_ct = water_data(coords)
        drVec = coords[pairs[1]] - coords[pairs[0]]

        # Dispersion parameters
        C6_disp, b_disp = param_disp

        disp_pairwise = computeDispersion(
            drVec, 
            C6_disp[pairs[0]], C6_disp[pairs[1]],
            b_disp[pairs[0]], b_disp[pairs[1]]
        )
        disp = torch.sum(disp_pairwise) / 2 * HARTREE2KCAL
        return disp
    
    energy = get_dispersion_energy(coords)
    energy.backward()
    grads_ad = coords.grad

    coords = get_water_dimer_coords(requires_grad=False)
    grads_fd = finite_difference(coords, get_dispersion_energy, h=1e-5)
    assert torch.allclose(grads_ad, grads_fd)

def test_xpol_gradients():
    torch.set_default_dtype(torch.float64)

    coords = get_water_dimer_coords()

    def get_xpol_energy(coords: torch.Tensor):
        pairs, param_elec, param_pauli, param_disp, param_pol, param_xpol, param_ct = water_data(coords)
        drVec = coords[pairs[1]] - coords[pairs[0]]

        # exchange-polarization parameters
        _, mPoles, _ = param_elec
        b_xpol, Kmono_xpol, Kdipo_xpol, Kquad_xpol = param_xpol
        mPoles_xpol = scaleMultipoles(mPoles, Kmono_xpol, Kdipo_xpol, Kquad_xpol)

        xpol_pairwise = computeShortRangeEnergy(
            drVec,
            mPoles_xpol[pairs[0]], mPoles_xpol[pairs[1]],
            b_xpol[pairs[0]], b_xpol[pairs[1]],
            False
        )
        xpol = torch.sum(xpol_pairwise) / 2 * HARTREE2KCAL
        return xpol
    
    energy = get_xpol_energy(coords)
    energy.backward()
    grads_ad = coords.grad

    coords = get_water_dimer_coords(requires_grad=False)
    grads_fd = finite_difference(coords, get_xpol_energy, h=1e-5)
    assert torch.allclose(grads_ad, grads_fd)

def test_pauli_gradients():
    torch.set_default_dtype(torch.float64)

    coords = get_water_dimer_coords()

    def get_pauli_energy(coords: torch.Tensor):
        pairs, param_elec, param_pauli, param_disp, param_pol, param_xpol, param_ct = water_data(coords)
        drVec = coords[pairs[1]] - coords[pairs[0]]

        # Pauli parameters
        _, mPoles, _ = param_elec
        b_pauli, Kmono_pauli, Kdipo_pauli, Kquad_pauli = param_pauli
        mPoles_pauli = scaleMultipoles(mPoles, Kmono_pauli, Kdipo_pauli, Kquad_pauli)
        pauli_pairwise = computeShortRangeEnergy(
            drVec,
            mPoles_pauli[pairs[0]], mPoles_pauli[pairs[1]],
            b_pauli[pairs[0]], b_pauli[pairs[1]]
        )
        pauli = torch.sum(pauli_pairwise) / 2 * HARTREE2KCAL
        return pauli
    
    energy = get_pauli_energy(coords)
    energy.backward()
    grads_ad = coords.grad

    coords = get_water_dimer_coords(requires_grad=False)
    grads_fd = finite_difference(coords, get_pauli_energy, h=1e-5)
    assert torch.allclose(grads_ad, grads_fd)

def test_ct_gradients():
    torch.set_default_dtype(torch.float64)

    coords = get_water_dimer_coords()

    def get_ct_energy(coords: torch.Tensor):
        pairs, param_elec, param_pauli, param_disp, param_pol, param_xpol, param_ct = water_data(coords)
        drVec = coords[pairs[1]] - coords[pairs[0]]

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
        dq_groups = scatter(dq, torch.tensor([0, 0, 0, 1, 1, 1]))

        _, pol = computePermElecAndPolarizationEnergy(
            coords,
            [[0, 1, 2], [3, 4, 5]],
            mPoles,
            Z,
            b,
            True,
            alpha,
            eta,
            groupCharges,
        )

        _, pol_ct = computePermElecAndPolarizationEnergy(
            coords,
            [[0, 1, 2], [3, 4, 5]],
            mPoles,
            Z,
            b,
            True,
            alpha,
            eta,
            groupCharges + dq_groups
        )
        pol *= HARTREE2KCAL
        pol_ct *= HARTREE2KCAL
        return ct_direct + (pol_ct - pol)
    
    energy = get_ct_energy(coords)
    energy.backward()
    grads_ad = coords.grad

    coords = get_water_dimer_coords(requires_grad=False)
    grads_fd = finite_difference(coords, get_ct_energy, h=1e-5)

    assert torch.allclose(grads_ad, grads_fd, atol=1e-7)
