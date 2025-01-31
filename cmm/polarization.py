import torch

from .electrostatics import computeInducedElectricPotentialAndFieldsFromPairs

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
    group_scatter: torch.Tensor,
    groups: torch.Tensor,
    residual_threshold: torch.Tensor = torch.tensor(1e-20),
    max_iter: torch.NumberType = 400,
):
    # TODO: This works well. We should always solve completely on the first step
    # and then provide an option for early stopping and/or to do autodiff back through
    # a few steps of CG (as described here: https://pubs.acs.org/doi/10.1021/acs.jctc.6b00981).
    # Basically just need to decide when/how we are going to do this. These options start
    # to interact with user setting though so will want to discuss with Eric.
    
    TM0, _, _ = computeProductWithPolarizationMatrix(
        guess_vector, n_charges,
        pairs_i_a, pairs_j_a,
        dists_p, dist_vecs_p,
        b_ij_p, eta, inverse_polarizabilities,
        group_scatter, groups
    )
    residual = b_vector - TM0
    P = residual.detach().clone()
    for _ in range(max_iter):
        TP, _, _ = computeProductWithPolarizationMatrix(
            P, n_charges,
            pairs_i_a, pairs_j_a,
            dists_p, dist_vecs_p,
            b_ij_p, eta, inverse_polarizabilities,
            group_scatter, groups
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
        #residual = residual - gamma * TP
        beta *= torch.dot(residual, residual)
        #beta = beta * torch.dot(residual, residual)
        P = residual + beta * P
        if torch.norm(residual) < residual_threshold:
            return guess_vector
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
    group_scatter: torch.Tensor,
    groups: torch.Tensor,
    already_solved: bool = False
):
    induced_charges = vec_in[:n_charges]
    induced_dipoles = vec_in[n_charges:(4 * n_charges)].view(-1, 3)
    lagrange_muls = vec_in[(4 * n_charges):]
    induced_multipoles_a = torch.hstack((induced_charges.unsqueeze(1), induced_dipoles))

    induced_electric_potential, induced_electric_field = computeInducedElectricPotentialAndFieldsFromPairs(
        n_charges, pairs_i_a, pairs_j_a,
        dists_p, dist_vecs_p, b_ij_p, induced_multipoles_a
    )

    constraints = torch.tensor([torch.sum(induced_charges[groups.unbind()[i]]) for i in torch.arange(groups.size(0))])
    
    res = torch.concat((eta * induced_charges + lagrange_muls[group_scatter] + induced_electric_potential, torch.bmm(alpha_inv, induced_dipoles.unsqueeze(-1)).squeeze(-1).flatten() - induced_electric_field.flatten(), constraints))
    return res, induced_electric_potential, induced_electric_field