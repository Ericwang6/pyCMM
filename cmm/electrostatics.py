from typing import List, Union, Optional
import torch
from torch_scatter import scatter

from .multipole import computeInteractionTensor, computeEletrostaticData


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


def computePolarizationDampFactors(dr, bij):
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

def computeElectricPotentialExpansion(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    mPoles: torch.Tensor,
    Z: torch.Tensor,
    b: torch.Tensor
):

    # expand mpoles
    mPoles_i, mPoles_j = mPoles[pairs[0]], mPoles[pairs[1]]


    drVec = coords[pairs[1]] - coords[pairs[0]]
    dr = torch.norm(drVec, dim=1)
    drInv = 1 / dr
    drInv3 = torch.pow(drInv, 3)
    drInv5 = torch.pow(drInv, 5)

    # damping factors
    b_i, b_j = b[pairs[0]], b[pairs[1]]
    b_ij = torch.sqrt(b_i * b_j)
    oneCenterDamps_i = computePermElecOneCenterDampFactors(dr, b_i)
    oneCenterDamps_j = computePermElecOneCenterDampFactors(dr, b_j)
    twoCenterDamps = computePermElecTwoCenterDampFactors(dr, b_ij)

    # NOTE(JOE): I am only taking one of the rows of pairs because this
    # list has both the forward and backward pairs in it.
    Z_pairs = Z[pairs[0]]
    # core-core interactions
    ePotCore = Z_pairs * drInv
    eFieldCore = drVec * (Z_pairs * drInv3).unsqueeze(-1)
    eFieldGradCore_1 = torch.vmap(torch.mul)(torch.vmap(torch.outer)(drVec, drVec), (3 * Z_pairs * drInv5))
    I = torch.eye(3)
    I = I.reshape((1, 3, 3))
    I = I.repeat((pairs[0].size(0), 1, 1))
    eFieldGradCore_2 = torch.vmap(torch.mul)(I, (Z_pairs * drInv3))
    eFieldGradCore = eFieldGradCore_2 - eFieldGradCore_1 # This is off by a minus sign for a reason I don't understand
    eFieldGradCore = eFieldGradCore.view(-1, 9)

    # core-shell interactions
    cs_tensor_ij = computeInteractionTensor(drVec, oneCenterDamps_i, drInv)
    cs_tensor_ji = computeInteractionTensor(-drVec, oneCenterDamps_j, drInv)

    cs_tensor_ij_0 = computeInteractionTensor(drVec, oneCenterDamps_i, drInv, rank=0)
    ePotCore_i = cs_tensor_ij_0 * Z_pairs

    eData_i = torch.bmm(cs_tensor_ij, mPoles_i.unsqueeze(2))
    eData_j = torch.bmm(cs_tensor_ji, mPoles_j.unsqueeze(2))
    ePot_i, ePot_j = eData_i[:, 0].flatten(), eData_j[:, 0].flatten()
    eField_i, eField_j = eData_i[:, 1:4].reshape(-1, 3), eData_j[:, 1:4].reshape(-1, 3)

    E_potentials = torch.zeros(coords.size(0)) # N
    E_fields = torch.zeros_like(coords) # Nx3
    E_field_grads = torch.zeros(coords.size(0), 9) # Nx3x3

    # @SPEED: This makes a copy because I accumulate into x, y, and z
    # separately. I am guessing there is a way to do this in one scatter
    # and reduce operation but I am not sure how. Just a bunch of reshaping?
    # One way to do it would be to expand the pairs array into indices into
    # the field array and similarly for field gradients array. Can just
    # precompute these I suppose.
    E_potentials.scatter_add_(0, pairs[1], ePot_i + ePotCore)
    E_fields[:, 0].scatter_add_(0, pairs[1], eFieldCore[:, 0] - eField_i[:, 0])
    E_fields[:, 1].scatter_add_(0, pairs[1], eFieldCore[:, 1] - eField_i[:, 1])
    E_fields[:, 2].scatter_add_(0, pairs[1], eFieldCore[:, 2] - eField_i[:, 2])
    # Yes, I am doing it like this. Please help.
    E_field_grads[:, 0].scatter_add_(0, pairs[1], eFieldGradCore[:, 0])
    E_field_grads[:, 1].scatter_add_(0, pairs[1], eFieldGradCore[:, 1])
    E_field_grads[:, 2].scatter_add_(0, pairs[1], eFieldGradCore[:, 2])
    E_field_grads[:, 3].scatter_add_(0, pairs[1], eFieldGradCore[:, 3])
    E_field_grads[:, 4].scatter_add_(0, pairs[1], eFieldGradCore[:, 4])
    E_field_grads[:, 5].scatter_add_(0, pairs[1], eFieldGradCore[:, 5])
    E_field_grads[:, 6].scatter_add_(0, pairs[1], eFieldGradCore[:, 6])
    E_field_grads[:, 7].scatter_add_(0, pairs[1], eFieldGradCore[:, 7])
    E_field_grads[:, 8].scatter_add_(0, pairs[1], eFieldGradCore[:, 8])

    #print(E_fields)
    #print(E_field_grads)

    # shell-shell interactions
    ss_tensor_ij = computeInteractionTensor(drVec, twoCenterDamps, drInv)
    ss_edata = torch.bmm(ss_tensor_ij, mPoles_i.unsqueeze(2))
    ePot_ss = ss_edata[:, 0].flatten()
    eField_ss = ss_edata[:, 1:4].reshape(-1, 3)

    #E_potentials_shell = torch.zeros(coords.size(0)) # N
    #E_fields_shell = torch.zeros_like(coords) # Nx3
    #E_field_grads_shell = torch.zeros(coords.size(0), 9) # Nx3x3
    #E_potentials_shell.scatter_add_(0, pairs[1], ePot_ss + ePotCore_i)
    #E_fields_shell[:, 0].scatter_add_(0, pairs[1], eField_i[:, 0])
    #E_fields_shell[:, 1].scatter_add_(0, pairs[1], eField_i[:, 1])
    #E_fields_shell[:, 2].scatter_add_(0, pairs[1], eField_i[:, 2])
    #print(E_potentials_shell)
    #print(E_fields_shell)
    #print(E_field_grads_shell)

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
        return elec, torch.tensor(0.0)
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

        polDamps_i = computePolarizationDampFactors(dr, b_ij)
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
        
        return elec, pol


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