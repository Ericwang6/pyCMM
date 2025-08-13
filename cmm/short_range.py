import torch
from .multipole import computeInteractionTensor
from typing import Optional


def computeShortRangeDampFactors(dr, bij):
    u = bij * dr
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    u6 = u5 * u
    u7 = u6 * u
    exp_u = torch.exp(-u)
    p1 = 1 + 11 * u / 16 + 3 * u2 / 16 + u3 / 48
    tmp = 1 + u + u2 / 2
    p3 = tmp + 7 * u3 / 48 + u4 / 48
    tmp += u3 / 6 +  u4 / 24 
    p5 = tmp + u5 / 144
    p7 = tmp + u5 / 120 + u6 / 720
    p9 = p7 + u7 / 5040

    return [p * exp_u for p in [p1, p3, p5, p7, p9]]


def scaleMultipoles(
    mPoles: torch.Tensor, 
    monoScales: torch.Tensor, dipoScales: torch.Tensor, quadScales: torch.Tensor
):
    # The monopoles are set directly from the parameter list while the
    # multipoles are directly scaled versions of the electric multipoles.
    mPolesScaled = torch.zeros_like(mPoles)
    mPolesScaled[:, 0]   += monoScales
    mPolesScaled[:, 1:4] += mPoles[:, 1:4] * dipoScales.unsqueeze(1)
    mPolesScaled[:, 4:]  += mPoles[:, 4:] * quadScales.unsqueeze(1)
    return mPolesScaled


def computeShortRangeEnergy(
    drVec: torch.Tensor,
    mPoles_i: torch.Tensor, mPoles_j: torch.Tensor,
    b_i: torch.Tensor, b_j: torch.Tensor,
    positive: bool = True,
):

    dr = torch.norm(drVec, dim=1)
    drInv = 1 / dr

    b_ij = torch.sqrt(b_i * b_j)
    damps = computeShortRangeDampFactors(dr, b_ij)
    if not positive:
        damps = [-d for d in damps]

    iTensor = computeInteractionTensor(drVec, damps, drInv, 2)
    enes = torch.bmm(mPoles_j.unsqueeze(1), torch.bmm(iTensor, mPoles_i.unsqueeze(2))).flatten()
    return enes

def computeShortRangeEnergyFromPairs(
    dists_p: torch.Tensor, dist_vecs_p: torch.Tensor,
    mPoles_i_p: torch.Tensor, mPoles_j_p: torch.Tensor,
    b_ij_p: torch.Tensor,
    positive: bool = True,
):

    drInv_p = 1 / dists_p

    damps = computeShortRangeDampFactors(dists_p, b_ij_p)
    if not positive:
        damps = [-d for d in damps]

    iTensor = computeInteractionTensor(dist_vecs_p, damps, drInv_p, 2)
    enes = torch.bmm(mPoles_j_p.unsqueeze(1), torch.bmm(iTensor, mPoles_i_p.unsqueeze(2))).flatten()
    return enes


def computePairwiseChargeTransfer(
    drVec: torch.Tensor,
    mPoles_acc_i: torch.Tensor, mPoles_acc_j: torch.Tensor,
    mPoles_don_i: torch.Tensor, mPoles_don_j: torch.Tensor,
    b_i: torch.Tensor, b_j: torch.Tensor,
    eps_ij: torch.Tensor
):
    dr = torch.norm(drVec, dim=1)
    drInv = 1 / dr

    b_ij = torch.sqrt(b_i * b_j)
    damps = computeShortRangeDampFactors(dr, b_ij)
    damps = [-d for d in damps]

    iTensor = computeInteractionTensor(drVec, damps, drInv, 2)
    enes_ij = torch.bmm(mPoles_don_j.unsqueeze(1), torch.bmm(iTensor, mPoles_acc_i.unsqueeze(2))).flatten()
    enes_ji = torch.bmm(mPoles_acc_j.unsqueeze(1), torch.bmm(iTensor, mPoles_don_i.unsqueeze(2))).flatten()
    enes = enes_ij + enes_ji

    # forward means i -> j, backward means j -> i
    drInvDamp = iTensor[:, 0, 0].flatten()
    dq_forward = mPoles_don_i[:, 0] * mPoles_acc_j[:, 0] * drInvDamp * eps_ij
    dq_backward = mPoles_acc_i[:, 0] * mPoles_don_j[:, 0] * drInvDamp * eps_ij
    return enes, dq_forward - dq_backward

def get_field_dependent_polarizabilities(
        polarizabilities_a: torch.Tensor,
        elec_field_a: torch.Tensor,
        alpha_damp_exponent_a: torch.Tensor,
        alpha_damp_max_a: torch.Tensor
    ):
    elec_field_mag_sq_a = torch.sum(elec_field_a * elec_field_a, dim=1)
    damp_factor_a = alpha_damp_max_a * (1 - torch.exp(-alpha_damp_exponent_a * elec_field_mag_sq_a))
    return polarizabilities_a - damp_factor_a.view(-1, 1, 1) * polarizabilities_a