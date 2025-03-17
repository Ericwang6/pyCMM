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
    pairs_lr_i_a: torch.Tensor, pairs_lr_j_a: torch.Tensor,
    pairs_sr_i_a: torch.Tensor, pairs_sr_j_a: torch.Tensor,
    pairs_excl_i_a: torch.Tensor, pairs_excl_j_a: torch.Tensor,
    direct_field_tensor_lr: torch.Tensor,
    pol_interaction_tensor_sr: torch.Tensor,
    direct_field_tensor_excl: torch.Tensor,
    induced_field_data: torch.Tensor,
    eta: torch.Tensor,
    inverse_polarizabilities: torch.Tensor,
    pol_group_indices_a: torch.Tensor,
    pol_group_segment_indices: torch.Tensor,
    pol_group_lengths_g: torch.Tensor,
    long_range_potential_function=None,
    residual_threshold: torch.Tensor = torch.tensor(1e-10),
    max_iter: torch.NumberType = 400,
):
    # TODO: This works well. We should always solve completely on the first step
    # and then provide an option for early stopping and/or to do autodiff back through
    # a few steps of CG (as described here: https://pubs.acs.org/doi/10.1021/acs.jctc.6b00981).
    # Basically just need to decide when/how we are going to do this. These options start
    # to interact with user setting though so will want to discuss with Eric.
    
    # Precondition using whatever was put in the guess_vector
    TM0, _, _  = computeProductWithPolarizationMatrix(
        guess_vector, n_charges,
        pairs_lr_i_a, pairs_lr_j_a,
        pairs_sr_i_a, pairs_sr_j_a,
        pairs_excl_i_a, pairs_excl_j_a,
        direct_field_tensor_lr, pol_interaction_tensor_sr, direct_field_tensor_excl,
        induced_field_data,
        eta, inverse_polarizabilities,
        pol_group_indices_a,
        pol_group_segment_indices,
        pol_group_lengths_g,
        long_range_potential_function
    )
    residual = b_vector - TM0
    P = residual.detach().clone()
    for i_iter in range(max_iter):
        TP, _, _ = computeProductWithPolarizationMatrix(
            P, n_charges,
            pairs_lr_i_a, pairs_lr_j_a,
            pairs_sr_i_a, pairs_sr_j_a,
            pairs_excl_i_a, pairs_excl_j_a,
            direct_field_tensor_lr, pol_interaction_tensor_sr, direct_field_tensor_excl,
            induced_field_data,
            eta, inverse_polarizabilities,
            pol_group_indices_a,
            pol_group_segment_indices,
            pol_group_lengths_g,
            long_range_potential_function
        )
        gamma = torch.dot(residual, residual) / torch.dot(P, TP)
        guess_vector += gamma * P
        beta = 1.0 / torch.dot(residual, residual)
        residual -= gamma * TP
        if torch.norm(residual) < residual_threshold:
            print(i_iter, " steps to converge")
            return guess_vector
        beta *= torch.dot(residual, residual)
        P = residual + beta * P
    return guess_vector

def computeProductWithPolarizationMatrix(
    vec_in: torch.Tensor,
    n_charges: torch.NumberType,
    pairs_lr_i_a: torch.Tensor, pairs_lr_j_a: torch.Tensor,
    pairs_sr_i_a: torch.Tensor, pairs_sr_j_a: torch.Tensor,
    pairs_excl_i_a: torch.Tensor, pairs_excl_j_a: torch.Tensor,
    direct_field_tensor_lr: torch.Tensor,
    pol_interaction_tensor_sr: torch.Tensor,
    direct_field_tensor_excl: torch.Tensor,
    induced_field_data: torch.Tensor,
    eta: torch.Tensor,
    alpha_inv: torch.Tensor,
    pol_group_indices_a: torch.Tensor,
    pol_group_segment_indices: torch.Tensor,
    pol_group_lengths_g: torch.Tensor,
    long_range_potential_function=None
):
    
    induced_charges = vec_in[:n_charges]
    induced_dipoles = vec_in[n_charges:(4 * n_charges)].view(-1, 3)
    lagrange_muls = vec_in[(4 * n_charges):]
    induced_multipoles_a = torch.hstack((induced_charges.unsqueeze(1), induced_dipoles))
    induced_multipoles_i_lr_p = induced_multipoles_a[pairs_lr_i_a]
    induced_multipoles_i_sr_p = induced_multipoles_a[pairs_sr_i_a]

    # Get real field data #
    edata_point_pairwise = torch.bmm(direct_field_tensor_lr, induced_multipoles_i_lr_p.unsqueeze(2))
    edata_ss_pairwise = torch.bmm(pol_interaction_tensor_sr, induced_multipoles_i_sr_p.unsqueeze(2))

    # Accumulate the total potentials, fields, and field gradients.
    # One contribution is accumulated over all long-range pairs, while the other
    # accumulates just the penetration contribution.
    induced_field_data = torch.zeros_like(induced_field_data) # <-- I think this avoids the allocation
    induced_field_data.scatter_add_(0, pairs_lr_j_a.unsqueeze(1).expand(-1, 4), edata_point_pairwise.squeeze(2))
    induced_field_data.scatter_add_(0, pairs_sr_j_a.unsqueeze(1).expand(-1, 4), edata_ss_pairwise.squeeze(2))
    
    if long_range_potential_function:
        induced_multipoles_i_excl_p = induced_multipoles_a[pairs_excl_i_a]
        edata_point_excl_pairwise = torch.bmm(direct_field_tensor_excl, induced_multipoles_i_excl_p.unsqueeze(2))
        induced_field_data.scatter_add_(0, pairs_excl_j_a.unsqueeze(1).expand(-1, 4), edata_point_excl_pairwise.squeeze(2))
    
    induced_field_data.mul_(torch.tensor([1, -1, -1, -1], device=pairs_lr_i_a.device).reshape(1, -1))
    induced_electric_potential = induced_field_data[:, 0]
    induced_electric_field = induced_field_data[:, 1:4]
    # Get reciprocal space field data (ewald + self contribution) #
    if long_range_potential_function:
        # Get reciprocal space and self contributions to field variables
        # and corresponding electrostatic interactions.
        ewald_potential, ewald_field = long_range_potential_function(induced_charges, induced_dipoles)
        induced_electric_potential = induced_electric_potential + ewald_potential
        induced_electric_field = induced_electric_field + ewald_field

    # Get sum of induced charges in every polarization group
    constraints = segment_csr(induced_charges[pol_group_indices_a], pol_group_segment_indices, reduce='sum')

    # Expand lagrange multipliers from group index space to atomic index space
    expanded_lagrange_muls = lagrange_muls.repeat_interleave(pol_group_lengths_g)

    # Now we need to scatter these values back to the atomic indices
    lagrange_muls_a = torch.zeros(n_charges, device=lagrange_muls.device)
    lagrange_muls_a.scatter_add_(0, pol_group_indices_a, expanded_lagrange_muls)

    res = torch.concat((
        eta * induced_charges + lagrange_muls_a + induced_electric_potential,
        torch.bmm(alpha_inv, induced_dipoles.unsqueeze(-1)).squeeze(-1).flatten() - induced_electric_field.flatten(),
        constraints
    ))
    return res, induced_electric_potential, induced_electric_field