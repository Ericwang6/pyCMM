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


def scaleMultipoles(mPoles, monoScales, dipoScales, quadScales):
    mPolesScaled = torch.zeros_like(mPoles)
    mPolesScaled[:, 0]   += mPoles[:, 0] * monoScales
    mPolesScaled[:, 1:4] += mPoles[:, 1:4] * dipoScales.unsqueeze(1)
    mPolesScaled[:, 4:]  += mPoles[:, 4:] * quadScales.unsqueeze(1)
    return mPolesScaled


def computeShortRangeEnergy(
    drVec: torch.Tensor,
    mPoles_i: torch.Tensor, mPoles_j: torch.Tensor,
    Kmono_i: torch.Tensor, Kmono_j: torch.Tensor,
    Kdipo_i: torch.Tensor, Kdipo_j: torch.Tensor,
    Kquad_i: torch.Tensor, Kquad_j: torch.Tensor,
    b_i: torch.Tensor, b_j: torch.Tensor,
    positive: bool = True,
    modCharge_i: Optional[torch.Tensor] = None, modCharge_j: Optional[torch.Tensor] = None
):
    mPolesShortRange_i = scaleMultipoles(mPoles_i, Kmono_i, Kdipo_i, Kquad_i)
    mPolesShortRange_j = scaleMultipoles(mPoles_j, Kmono_j, Kdipo_j, Kquad_j)

    if modCharge_i is not None:
        mPolesShortRange_i[:, 0] += modCharge_i
    if modCharge_j is not None:
        mPolesShortRange_j[:, 0] += modCharge_j

    dr = torch.norm(drVec, dim=1)
    drInv = 1 / dr

    b_ij = torch.sqrt(b_i * b_j)
    damps = computeShortRangeDampFactors(dr, b_ij)
    if not positive:
        damps = [-d for d in damps]

    iTensor = computeInteractionTensor(drVec, damps, drInv, 2)
    enes = torch.bmm(mPolesShortRange_j.unsqueeze(1), torch.bmm(iTensor, mPolesShortRange_i.unsqueeze(2))).flatten()
    return enes