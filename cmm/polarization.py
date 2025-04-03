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
    n_groups: torch.NumberType,
    polarizabilities: torch.Tensor,
    elec_field: torch.Tensor
):
    dipole_part = torch.bmm(polarizabilities, elec_field.unsqueeze(-1)).squeeze(-1).flatten()

    return torch.cat([
        torch.zeros(n_charges, device=elec_field.device),
        dipole_part,
        torch.zeros(n_groups, device=elec_field.device)
    ])

def compute_product_with_polarization_matrix(
    vec_in: torch.Tensor, n_charges: torch.Tensor,
    pairs_lr_i_a: torch.Tensor, pairs_lr_j_a: torch.Tensor,
    pairs_sr_i_a: torch.Tensor, pairs_sr_j_a: torch.Tensor,
    pairs_excl_i_a: torch.Tensor, pairs_excl_j_a: torch.Tensor,
    direct_field_tensor_lr: torch.Tensor,
    pol_interaction_tensor_sr: torch.Tensor,
    direct_field_tensor_excl: torch.Tensor,
    eta: torch.Tensor, alpha_inv: torch.Tensor,
    pol_group_indices_a: torch.Tensor,
    pol_group_segment_indices: torch.Tensor,
    pol_group_lengths_g: torch.Tensor,
    long_range_potential_function=None):

        induced_charges = torch.narrow(vec_in, 0, 0, n_charges)
        induced_dipoles = torch.narrow(vec_in, 0, n_charges, 3 * n_charges).reshape(n_charges, 3)
        lagrange_muls = torch.narrow(vec_in, 0, n_charges + 3 * n_charges, vec_in.size(0) - n_charges - 3 * n_charges)
        induced_multipoles_a = torch.cat([induced_charges.unsqueeze(1), induced_dipoles], dim=1)

        induced_multipoles_i_lr_p = induced_multipoles_a[pairs_lr_i_a]
        induced_multipoles_i_sr_p = induced_multipoles_a[pairs_sr_i_a]

        # Get real field data
        edata_point_pairwise = torch.bmm(direct_field_tensor_lr, induced_multipoles_i_lr_p.unsqueeze(2))
        edata_ss_pairwise = torch.bmm(pol_interaction_tensor_sr, induced_multipoles_i_sr_p.unsqueeze(2))

        # Accumulate the total potentials and fields
        induced_field_data = torch.zeros(n_charges, 4, device=induced_multipoles_a.device, dtype=induced_multipoles_a.dtype, requires_grad=True)
        induced_field_data = induced_field_data.scatter_add(0, pairs_lr_j_a.unsqueeze(1).expand(-1, 4), edata_point_pairwise.squeeze(2))
        induced_field_data = induced_field_data.scatter_add(0, pairs_sr_j_a.unsqueeze(1).expand(-1, 4), edata_ss_pairwise.squeeze(2))
        
        if long_range_potential_function:
            induced_multipoles_i_excl_p = induced_multipoles_a[pairs_excl_i_a]
            edata_point_excl_pairwise = torch.bmm(direct_field_tensor_excl, induced_multipoles_i_excl_p.unsqueeze(2))
            induced_field_data = induced_field_data.scatter_add(0, pairs_excl_j_a.unsqueeze(1).expand(-1, 4), edata_point_excl_pairwise.squeeze(2))
        
        induced_field_data = induced_field_data.mul(torch.tensor([1, -1, -1, -1], device=pairs_lr_i_a.device).reshape(1, -1))
        induced_electric_potential = induced_field_data[:, 0]
        induced_electric_field = induced_field_data[:, 1:4]

        # Get reciprocal space field data (ewald + self contribution)
        if long_range_potential_function:
            ewald_potential, ewald_field = long_range_potential_function(induced_charges, induced_dipoles)
            induced_electric_potential = induced_electric_potential + ewald_potential
            induced_electric_field = induced_electric_field + ewald_field

        # Get sum of induced charges in every polarization group
        constraints = segment_csr(induced_charges[pol_group_indices_a], pol_group_segment_indices, reduce='sum')

        # Expand lagrange multipliers from group index space to atomic index space
        expanded_lagrange_muls = lagrange_muls.repeat_interleave(pol_group_lengths_g)

        # Scatter these values back to the atomic indices
        lagrange_muls_a = torch.zeros(n_charges, device=lagrange_muls.device, requires_grad=True)
        lagrange_muls_a = lagrange_muls_a.scatter_add(0, pol_group_indices_a, expanded_lagrange_muls)

        res = torch.concat((
            eta * induced_charges + lagrange_muls_a + induced_electric_potential,
            torch.bmm(alpha_inv, induced_dipoles.unsqueeze(-1)).squeeze(-1).flatten() - induced_electric_field.flatten(),
            constraints
        ))
        return res, induced_electric_potential, induced_electric_field