import torch

def charge_flux_bond(
    dR: torch.Tensor, j_cf: torch.Tensor, r_e: torch.Tensor,
    
)

def charge_flux_around_central_atom(
    dR: torch.Tensor, theta: torch.Tensor,
    j_cf: torch.Tensor, j_bb_cf: torch.Tensor, j_theta: torch.Tensor,
    r_e: torch.Tensor, theta_e: torch.Tensor
) -> torch.Tensor:
    """
    Computes the change in charge due to charge flux.
    Need to ensure that the vector differences are always computed
    in the correct manner. I think this will be enforced by matching
    the pair of atom types with the ordering of the atoms or something
    like that?
    """
    #dq = (
    #    j_cf * (dR - r_e) +
    #    j_cf_bb * ()
    #)

def field_dependent_morse_params(
        dR_vec: torch.Tensor, dR: torch.Tensor, E: torch.Tensor,
        k_e: torch.Tensor, D_e: torch.Tensor, r_e: torch.Tensor,
        dipole_1: torch.Tensor, dipole_2: torch.Tensor,
        ct_slope_1: torch.Tensor, ct_slope_2: torch.Tensor, dQ_ct: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the field-dependent force constants and equilibrium distances
    used in the CMM Morse potential. All of these tensors should have N_fd_bond
    entries where N_fd_bond is the number of bonds which are field-dependent.
    Note that dR_vec gets dotted with E, so you need to ensure that the distance
    vectors are computed in the right direction.
    """
    E_proj = torch.func.vmap(torch.dot)(dR_vec, E) / dR
    dr_e = E_proj * dipole_1 / (k_e - E_proj * dipole_2) + ct_slope_1 * dQ_ct * dQ_ct
    k_e_fd = k_e - (3 * k_e * torch.sqrt(0.5 * k_e / D_e) * dr_e + E_proj * dipole_2) + ct_slope_2 * dQ_ct * dQ_ct
    
    # Ideally this will never happen but this is how I implemented it originally
    # to avoid the possiblity of taking a sqrt of a negative force constant
    # during the energy evaluation. Really hitting this branch indicates
    # the field is too strong for this model to be reasonable or that the
    # parameters determining the change in force constant are unrealistic.
    k_e_fd = torch.clamp(k_e_fd, 0.4 * k_e)
    return (r_e + dr_e, k_e_fd)

if __name__ == "__main__":
    R = 