import os 
import torch
from typing import List
import random


def fill_A_b_old(
    numSites: int, numGroups: int, 
    eta: torch.Tensor, # (numSites,)
    alpha_inv: torch.Tensor, # (numSites, 3, 3)
    groups: List[List[int]], # (numGroups length)
    pairs: torch.Tensor, # (2, n_pairs)
    polTensor: torch.Tensor, # (n_pairs, 4, 4)
    ePot: torch.Tensor, # (numSites,)
    eField: torch.Tensor, #(numSites, 3)
    groupCharges: torch.Tensor #(numGroups,)
):
    # fill A matrix
    dimA = numSites + numGroups + numSites * 3
    matA = torch.zeros((dimA, dimA))

    numRange = torch.arange(numSites)
    # diag qq - hardness
    matA[numRange, numRange] += eta
    # diag dd - inv polarizabilities
    offset = numSites + numGroups
    for i in range(numSites):
        matA[i*3+offset:(i+1)*3+offset, i*3+offset:(i+1)*3+offset] += alpha_inv[i]
    # charge conservation within groups
    for i in range(numGroups):
        matA[numSites + i, groups[i]] = 1.0
        matA[groups[i], numSites + i] = 1.0

    for i, (ai, aj) in enumerate(zip(pairs[0], pairs[1])):
        # dipo-dipo 
        matA[aj*3+offset: (aj+1)*3+offset, ai*3+offset: (ai+1)*3+offset] += polTensor[i, -3:, -3:]
        # charge-charge
        matA[aj, ai] += polTensor[i, 0, 0]
        # charge-dipo
        matA[aj, ai*3+offset:(ai+1)*3+offset] += polTensor[i, 0, -3:]
        matA[aj*3+offset:(aj+1)*3+offset, ai] += polTensor[i, -3:, 0]
    
    # solution vector
    vecB = torch.hstack((-ePot, groupCharges, eField.flatten())).unsqueeze(1)
    vecSolution = torch.linalg.solve(matA, vecB)
    pol = torch.matmul(vecSolution.T, (0.5 * torch.matmul(matA, vecSolution) - vecB)).squeeze()
    return matA, vecB, vecSolution, pol



def fill_A_b_new(
    natoms: int, n_pol_groups: int, 
    eta_times_2: torch.Tensor, # (nbz*numSites,)
    inverse_polarizabilities: torch.Tensor, # (nbz*numSites, 3, 3)
    atom_indices_to_group_indices: torch.Tensor, #(numSites, )
    all_pairs: torch.Tensor, # (n_pairs, 2) / not bi-direction
    pol_tensor: torch.Tensor, # (n_pairs, 4, 4)
    epot: torch.Tensor, # (numSites*nbz,)
    efield: torch.Tensor, #(numSites*nbz, 3)
    dq_groups: torch.Tensor #(nbz, numGroups)
):
    nbz = dq_groups.shape[0]
    all_pairs_i, all_pairs_j = all_pairs[:, 0], all_pairs[:, 1]
    atom_indices = torch.arange(natoms)
    _row_indices_1x1 = atom_indices * 4

    _col_indices_4x4_raw = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    _row_indices_4x4_raw = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    
    _row_indices_4x4 = torch.flatten(_row_indices_4x4_raw.expand(all_pairs.shape[0], -1) + all_pairs_j.reshape(-1, 1) * 4)
    _col_indices_4x4 = torch.flatten(_col_indices_4x4_raw.expand(all_pairs.shape[0], -1) + all_pairs_i.reshape(-1, 1) * 4)

    _row_indices_4x4_transpose = torch.flatten(_row_indices_4x4_raw.expand(all_pairs.shape[0], -1) + all_pairs_i.reshape(-1, 1) * 4)
    _col_indices_4x4_transpose = torch.flatten(_col_indices_4x4_raw.expand(all_pairs.shape[0], -1) + all_pairs_j.reshape(-1, 1) * 4)

    _row_indices_3x3 = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    _col_indices_3x3 = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])

    _row_indices_3x3 = torch.flatten(_row_indices_3x3.expand(natoms, -1) + _row_indices_1x1.reshape(-1, 1) + 1)
    _col_indices_3x3 = torch.flatten(_col_indices_3x3.expand(natoms, -1) + _row_indices_1x1.reshape(-1, 1) + 1)
    _row_indices_constraint = atom_indices_to_group_indices + natoms * 4
    _col_indices_constraint = _row_indices_1x1

    _fill_bvec_epot_indices = _row_indices_1x1
    _fill_bvec_efield_indices = torch.flatten(torch.tensor([1, 2, 3]).expand(natoms, -1) + _row_indices_1x1.reshape(-1, 1))


    b_vector = torch.zeros((nbz, natoms*4+n_pol_groups))
    b_vector[:, _fill_bvec_epot_indices] = -epot.reshape(nbz, -1)
    b_vector[:, _fill_bvec_efield_indices] = efield.reshape(nbz, -1)
    
    # fill A-matrix
    A_matrix = torch.zeros((nbz, n_pol_groups+natoms*4, n_pol_groups+natoms*4))
    A_matrix[:, _row_indices_1x1, _row_indices_1x1] = eta_times_2.reshape(nbz, -1)
    A_matrix[:, _row_indices_3x3, _col_indices_3x3] = inverse_polarizabilities.reshape(nbz, -1)
    A_matrix[:, _row_indices_4x4, _col_indices_4x4] = pol_tensor.reshape(nbz, -1)
    A_matrix[:, _row_indices_4x4_transpose, _col_indices_4x4_transpose] = pol_tensor.permute(0, 2, 1).reshape(nbz, -1)
    A_matrix[:, _row_indices_constraint, _col_indices_constraint] = torch.ones((nbz, natoms))
    A_matrix[:, _col_indices_constraint, _row_indices_constraint] = torch.ones((nbz, natoms))
    solutions = torch.linalg.solve(A_matrix, b_vector)
    ene_pol = torch.bmm(solutions.unsqueeze(1), 0.5 * torch.bmm(A_matrix, solutions.unsqueeze(2)) - b_vector.unsqueeze(2)).squeeze()

    return A_matrix, b_vector, solutions, ene_pol


def test_pol_mat_construct():
    natoms = 6
    ngroups = 2
    pairs = torch.tensor([[0,3],[0,4],[0,5],[1,3],[1,4],[1,5],[2,3],[2,4],[2,5]])
    eta = torch.rand(natoms)
    alpha_inv = torch.zeros((natoms, 3, 3))
    alpha_inv[:, torch.arange(3), torch.arange(3)] = torch.rand(natoms, 3)
    epot = torch.rand(natoms)
    efield = torch.rand(natoms, 3)
    
    pol_tensor = []
    for i in range(len(pairs)):
        drInv = random.random()
        tx, ty, tz = random.random(), random.random(), random.random()
        txx, txy, txz, tyy, tyz, tzz = random.random(), random.random(), random.random(), random.random(), random.random(), random.random()

        pol_tensor.append([
            [drInv, -tx,   -ty,   -tz],   
            [tx,    -txx,  -txy,  -txz],  
            [ty,    -txy,  -tyy,  -tyz],  
            [tz,    -txz,  -tyz,  -tzz],  
        ])
    pol_tensor = torch.tensor(pol_tensor)

    groups = [[0, 1, 2], [3, 4, 5]]
    groupCharges = torch.tensor([0.0, 0.0])

    pairs_old = torch.vstack((pairs, pairs[:, [1, 0]])).T
    pol_tensor_old = torch.vstack((pol_tensor, pol_tensor.permute(0, 2, 1)))
    result_old = fill_A_b_old(
        natoms, ngroups, eta, alpha_inv, groups, 
        pairs_old, pol_tensor_old, epot, efield, groupCharges
    )

    eta_bz = torch.concat((eta, eta))
    alpha_inv_bz = torch.concat((alpha_inv, alpha_inv))
    epot_bz = torch.cat((epot, epot))
    efield_bz = torch.cat((efield, efield))
    dq_groups_bz = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    atom_indices_to_group_indices = torch.tensor([0, 0, 0, 1, 1, 1])
    pol_tensor_bz = torch.concat((pol_tensor, pol_tensor))

    result_new = fill_A_b_new(
        natoms, ngroups, eta_bz, alpha_inv_bz, atom_indices_to_group_indices, 
        pairs, pol_tensor_bz, epot_bz, efield_bz, dq_groups_bz
    )

    assert result_old[-1].item() == result_new[-1][0].item()
    assert result_new[-1][1].item() == result_new[-1][0].item()