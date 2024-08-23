from typing import List, Union, Optional
import torch
from torch_scatter import scatter
from .multipole import computeInteractionTensor


def computePermElecOneCenterDampFactors(dr, b):
    u = b * dr
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    exp_u = torch.exp(-u)
    p1 = 1 + u / 2
    p3 = 1 + u + u2 / 2
    p5 = p3 + u3 / 6
    p7 = p5 + u4 / 30
    p9 = p5 + u4 * 4 / 105 + u5 / 210

    return [1 - p * exp_u for p in [p1, p3, p5, p7, p9]]


def computePermElecTwoCenterDampFactors(dr, bij):
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

    return [1 - p * exp_u for p in [p1, p3, p5, p7, p9]]


def computePolairzationDampFactors(dr, bij):
    u = bij * dr
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    u6 = u5 * u
    exp_u = torch.exp(-u)
    p1 = 1 + 1/9 * u + 1/11 * u2 + 1/13 * u3 + 1/15 * u4
    p3 = 1 + u + 2/99 * u2 - 9/143 * u3 - 8/65 * u4 + 1/15 * u5
    p5 = 1 + u + 101/297 * u2 + 2/197 * u3 - 43/2145 * u4 - 10/117 * u5 + 1/45 * u6
    return [1 - p * exp_u for p in [p1, p3, p5]]


def getPairsFromGroups(groups: List[List[int]]):
    pairs = []
    for gi in range(len(groups)):
        for gj in range(gi + 1, len(groups)):
            for ai in groups[gi]:
                for aj in groups[gj]:
                    pairs.append([ai, aj])
                    pairs.append([aj, ai])
    pairs = torch.tensor(pairs).T
    return pairs


def computePermElecAndPolarizationEnergy(
    coords: torch.Tensor,
    groups: List[List[int]],
    mPoles: torch.Tensor,
    Z: torch.Tensor,
    b: torch.Tensor,
    doPolarization: bool = True,
    alpha: Optional[torch.Tensor] = None,
    eta: Optional[torch.Tensor] = None,
    groupCharges: Optional[torch.Tensor] = None,
    groupChargesCT: Optional[torch.Tensor] = None,
    pairs: Optional[torch.Tensor] = None
):
    numSites = coords.shape[0]
    numGroups = len(groups)

    if pairs is None:
        pairs = getPairsFromGroups(groups)

    # expand mpoles
    mPoles_i, mPoles_j = mPoles[pairs[0]], mPoles[pairs[1]]

    drVec = coords[pairs[1]] - coords[pairs[0]]
    dr = torch.norm(drVec, dim=1)
    drInv = 1 / dr

    # damping factors
    b_i, b_j = b[pairs[0]], b[pairs[1]]
    b_ij = torch.sqrt(b_i * b_j)
    oneCenterDamps_i = computePermElecOneCenterDampFactors(dr, b_i)
    oneCenterDamps_j = computePermElecOneCenterDampFactors(dr, b_j)
    twoCenterDamps = computePermElecTwoCenterDampFactors(dr, b_ij)

    Z_i, Z_j = Z[pairs[0]], Z[pairs[1]]
    # core-core interactions
    ePotCore_i = Z_i * drInv
    ccPairwiseEnergies = Z_j * ePotCore_i

    # core-shell interactions
    cs_tensor_ij = computeInteractionTensor(drVec, oneCenterDamps_i, drInv)
    cs_tensor_ji = computeInteractionTensor(-drVec, oneCenterDamps_j, drInv)

    eData_i = torch.bmm(cs_tensor_ij, mPoles_i.unsqueeze(2))
    eData_j = torch.bmm(cs_tensor_ji, mPoles_j.unsqueeze(2))
    ePot_i, ePot_j = eData_i[:, 0].flatten(), eData_j[:, 0].flatten()
    scPairwiseEnergies = ePot_i * Z_j + ePot_j * Z_i

    # shell-shell interactions
    ss_tensor_ij = computeInteractionTensor(drVec, twoCenterDamps, drInv)
    ss_edata = torch.bmm(ss_tensor_ij, mPoles_i.unsqueeze(2))
    ssPairwiseEnergies = torch.bmm(mPoles_j.unsqueeze(1), ss_edata).flatten()

    elecPairwiseEnergies = ccPairwiseEnergies + scPairwiseEnergies + ssPairwiseEnergies
    elec = torch.sum(elecPairwiseEnergies) / 2

    if not doPolarization:
        return elec, torch.tensor(0.0), torch.tensor(0.0)
    else:
        # fill B vector
        ePotCore = scatter(ePotCore_i + ePot_i, pairs[1])
        # electric field by core charges
        eFieldCore_i = drVec * torch.unsqueeze(Z_i * torch.pow(drInv, 3), 1)
        # plus electric field by damped multipoles
        eField_i = eFieldCore_i - eData_i[:, 1:4].squeeze(2)
        eField = scatter(eField_i, pairs[1], dim=0)
        vecB = torch.hstack((-ePotCore, groupCharges, eField.flatten())).unsqueeze(1)

        # fill A matrix
        dimA = numSites + numGroups + numSites * 3
        matA = torch.zeros((dimA, dimA))

        numRange = torch.arange(numSites)
        # diag qq - hardness
        matA[numRange, numRange] += eta
        # diag dd - inv polarizabilities
        alpha_inv = torch.linalg.inv(alpha)
        offset = numSites + numGroups
        for i in range(numSites):
            matA[i*3+offset:(i+1)*3+offset, i*3+offset:(i+1)*3+offset] += alpha_inv[i]
        # charge conservation within groups
        for i in range(numGroups):
            matA[numSites + i, groups[i]] = 1.0
            matA[groups[i], numSites + i] = 1.0
        
        polDamps_i = computePolairzationDampFactors(dr, b_ij)
        polTensor = computeInteractionTensor(drVec, polDamps_i, rank=1)
        for i, (ai, aj) in enumerate(zip(pairs[0], pairs[1])):
            # dipo-dipo 
            matA[aj*3+offset: (aj+1)*3+offset, ai*3+offset: (ai+1)*3+offset] += polTensor[i, -3:, -3:]
            # charge-charge
            matA[aj, ai] += polTensor[i, 0, 0]
            # charge-dipo
            matA[aj, ai*3+offset:(ai+1)*3+offset] += polTensor[i, 0, -3:]
            matA[aj*3+offset:(aj+1)*3+offset, ai] += polTensor[i, -3:, 0]
        
        # solution vector
        vecSolution = torch.matmul(torch.linalg.inv(matA), vecB)
        pol = torch.matmul(vecSolution.T, (0.5 * torch.matmul(matA, vecSolution) - vecB)).squeeze()

        if groupChargesCT is not None:
            vecB_ct = torch.hstack((-ePotCore, groupCharges + groupChargesCT, eField.flatten())).unsqueeze(1)
            vecSolution_ct = torch.matmul(torch.linalg.inv(matA), vecB_ct)
            pol_ct = torch.matmul(vecSolution_ct.T, (0.5 * torch.matmul(matA, vecSolution_ct) - vecB_ct)).squeeze()
            return elec, pol, pol_ct
        else:
            return elec, pol, torch.tensor(0.0)


if __name__ == '__main__':
    coords = torch.rand(6, 3)
    groups = [[0, 1, 2], [3, 4, 5]]
    groupCharges = torch.tensor([0.0, 0.0])
    alpha = torch.rand(6, 3, 3)
    eta = torch.rand(6)
    matA, vecB = computePermElecAndPolarizationEnergy(coords, groups, groupCharges, alpha, eta)
    print(alpha[-1])
    print(matA[-3:,-3:])
    print(vecB)