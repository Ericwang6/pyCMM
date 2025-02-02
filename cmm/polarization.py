import torch
from torch_scatter import segment_csr
from typing import Optional

from .electrostatics import computeInducedElectricPotentialAndFieldsFromPairs

def get_field_dependent_polarizabilities(
        polarizabilities_a: torch.Tensor,
        elec_field_a: torch.Tensor,
        alpha_damp_exponent_a: torch.Tensor,
        alpha_damp_max_a: torch.Tensor
    ):
    elec_field_mag_sq_a = torch.func.vmap(torch.dot)(elec_field_a, elec_field_a)
    damp_factor_a = alpha_damp_max_a * (1 - torch.exp(-alpha_damp_exponent_a * elec_field_mag_sq_a))
    return polarizabilities_a - damp_factor_a.view(-1, 1, 1) * polarizabilities_a

def direct_field_induced_dipole_guess(
    n_charges: torch.NumberType,
    n_dipoles: torch.NumberType,
    n_groups: torch.NumberType,
    polarizabilities: torch.Tensor,
    elec_field: torch.Tensor
):
    guess_vector = torch.zeros(n_charges + 3 * n_dipoles + n_groups, device=elec_field.device)
    guess_vector[n_charges:(n_charges + 3 * n_dipoles)] = torch.bmm(polarizabilities, elec_field.unsqueeze(-1)).squeeze(-1).flatten()
    return guess_vector

def solvePolarizationByCG(
    guess_vector: torch.Tensor,
    b_vector: torch.Tensor,
    n_charges: torch.NumberType,
    pairs_i_a: torch.Tensor,
    pairs_j_a: torch.Tensor,
    dists_p: torch.Tensor,
    dist_vecs_p: torch.Tensor,
    b_ij_p: torch.Tensor,
    eta: torch.Tensor,
    inverse_polarizabilities: torch.Tensor,
    pol_group_indices_a: torch.Tensor,
    pol_group_segment_indices: torch.Tensor,
    pol_group_lengths_g: torch.Tensor,
    residual_threshold: torch.Tensor = torch.tensor(1e-20),
    max_iter: torch.NumberType = 400,
):
    # TODO: This works well. We should always solve completely on the first step
    # and then provide an option for early stopping and/or to do autodiff back through
    # a few steps of CG (as described here: https://pubs.acs.org/doi/10.1021/acs.jctc.6b00981).
    # Basically just need to decide when/how we are going to do this. These options start
    # to interact with user setting though so will want to discuss with Eric.
    
    # Precondition using whatever was put in the guess_vector
    TM0, _, _ = computeProductWithPolarizationMatrix(
        guess_vector, n_charges,
        pairs_i_a, pairs_j_a,
        dists_p, dist_vecs_p,
        b_ij_p, eta, inverse_polarizabilities,
        pol_group_indices_a,
        pol_group_segment_indices,
        pol_group_lengths_g
    )
    residual = b_vector - TM0
    P = residual.detach().clone()
    for _ in range(max_iter):
        TP, _, _ = computeProductWithPolarizationMatrix(
            P, n_charges,
            pairs_i_a, pairs_j_a,
            dists_p, dist_vecs_p,
            b_ij_p, eta, inverse_polarizabilities,
            pol_group_indices_a,
            pol_group_segment_indices,
            pol_group_lengths_g
        )
        gamma = torch.dot(residual, residual) / torch.dot(P, TP)
        # NOTE(JOE): When we switch to not using autodiff through
        # the polarization solver, then we can uncomment the below
        # line and remove the one after which is needed for autodiff
        # tracking to work. After doing so, this function should return
        # nothing and we can just use the guess vector as the solution
        # vector since it gets updated in place.
        guess_vector += gamma * P
        #guess_vector = guess_vector + gamma * P
        beta = 1.0 / torch.dot(residual, residual)
        residual -= gamma * TP
        if torch.norm(residual) < residual_threshold:
            return guess_vector
        #residual = residual - gamma * TP
        beta *= torch.dot(residual, residual)
        #beta = beta * torch.dot(residual, residual)
        P = residual + beta * P
    return guess_vector

def computeProductWithPolarizationMatrix(
    vec_in: torch.Tensor,
    n_charges: torch.NumberType,
    pairs_i_a: torch.Tensor,
    pairs_j_a: torch.Tensor,
    dists_p: torch.Tensor,
    dist_vecs_p: torch.Tensor,
    b_ij_p: torch.Tensor,
    eta: torch.Tensor,
    alpha_inv: torch.Tensor,
    pol_group_indices_a: torch.Tensor,
    pol_group_segment_indices: torch.Tensor,
    pol_group_lengths_g: torch.Tensor
):
    induced_charges = vec_in[:n_charges]
    induced_dipoles = vec_in[n_charges:(4 * n_charges)].view(-1, 3)
    lagrange_muls = vec_in[(4 * n_charges):]
    induced_multipoles_a = torch.hstack((induced_charges.unsqueeze(1), induced_dipoles))

    induced_electric_potential, induced_electric_field = computeInducedElectricPotentialAndFieldsFromPairs(
        n_charges, pairs_i_a, pairs_j_a,
        dists_p, dist_vecs_p, b_ij_p, induced_multipoles_a
    )

    # Get sum of induced charges in every polarization group
    constraints = segment_csr(induced_charges[pol_group_indices_a], pol_group_segment_indices, reduce='sum')

    # Expand lagrange multipliers from group index space to atomic index space
    expanded_lagrange_muls = lagrange_muls.repeat_interleave(pol_group_lengths_g)

    # Now we need to scatter these values back to the atomic indices
    lagrange_muls_a = torch.zeros(n_charges, device=lagrange_muls.device)
    lagrange_muls_a.scatter_add_(0, pol_group_indices_a, expanded_lagrange_muls)

    #lagrange_muls_a = lagrange_muls[group_scatter]
    res = torch.concat((eta * induced_charges + lagrange_muls_a + induced_electric_potential, torch.bmm(alpha_inv, induced_dipoles.unsqueeze(-1)).squeeze(-1).flatten() - induced_electric_field.flatten(), constraints))
    return res, induced_electric_potential, induced_electric_field