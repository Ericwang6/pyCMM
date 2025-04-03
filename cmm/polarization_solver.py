import torch
from .polarization import compute_product_with_polarization_matrix

class PolarizationSolver:
    def __init__(self, max_iter=400, tol=1e-7, solver_type="conjugate_gradient"):
        """
        Initialize the PolarizationSolver.
        
        Args:
            max_iter: Maximum number of iterations for the solver
            tol: Convergence tolerance
            solver_type: "conjugate_gradient" is only option for now
        """
        self.iterations_to_solve = 0
        self.max_iter = max_iter
        self.tol = tol
        self.solver_type = solver_type
        self.last_induced_multipoles = None

    def solve_by_conjugate_gradient(self, initial_guess, b_vector, n_charges,
                           pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                           pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                           pol_interaction_tensor_sr, direct_field_tensor_excl,
                           eta, inverse_polarizabilities, pol_group_indices_a,
                           pol_group_segment_indices, pol_group_lengths_g,
                           long_range_potential_function=None):
        # Precondition using whatever was put in the guess_vector
        TM0, _, _ = compute_product_with_polarization_matrix(
            initial_guess, n_charges,
            pairs_lr_i_a, pairs_lr_j_a,
            pairs_sr_i_a, pairs_sr_j_a,
            pairs_excl_i_a, pairs_excl_j_a,
            direct_field_tensor_lr, pol_interaction_tensor_sr, direct_field_tensor_excl,
            eta, inverse_polarizabilities,
            pol_group_indices_a,
            pol_group_segment_indices,
            pol_group_lengths_g,
            long_range_potential_function
        )
        residual = b_vector - TM0
        P = residual.clone()
        solution_vector = initial_guess.clone()
        
        for i_iter in range(self.max_iter):
            TP, _, _ = compute_product_with_polarization_matrix(
                P, n_charges,
                pairs_lr_i_a, pairs_lr_j_a,
                pairs_sr_i_a, pairs_sr_j_a,
                pairs_excl_i_a, pairs_excl_j_a,
                direct_field_tensor_lr, pol_interaction_tensor_sr, direct_field_tensor_excl,
                eta, inverse_polarizabilities,
                pol_group_indices_a,
                pol_group_segment_indices,
                pol_group_lengths_g,
                long_range_potential_function
            )
            gamma = torch.dot(residual, residual) / torch.dot(P, TP)
            solution_vector = solution_vector + gamma * P
            beta = 1.0 / torch.dot(residual, residual)
            residual = residual - gamma * TP
            if torch.norm(residual) < self.tol:
                self.iterations_to_solve = i_iter
                return solution_vector
            beta = beta * torch.dot(residual, residual)
            P = residual + beta * P
        
        return solution_vector
    
    def solve(self, initial_guess, elec_potential, elec_field, dq_groups, n_charges,
            pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
            pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
            pol_interaction_tensor_sr, direct_field_tensor_excl,
            eta, inverse_polarizabilities, pol_group_indices_a,
            pol_group_segment_indices, pol_group_lengths_g,
            long_range_potential_function=None):
        
        # Solve the system using the selected method
        b_vector = torch.hstack((-elec_potential, elec_field.flatten(), dq_groups))
        with torch.no_grad():
            if self.solver_type == "conjugate_gradient":
                solution = self.solve_by_conjugate_gradient(
                    initial_guess, b_vector, n_charges,
                    pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                    pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                    pol_interaction_tensor_sr, direct_field_tensor_excl,
                    eta, inverse_polarizabilities, pol_group_indices_a,
                    pol_group_segment_indices, pol_group_lengths_g,
                    long_range_potential_function
                )

        # Cache the solution for next time
        self.last_induced_multipoles = solution.clone().detach()

        # Compute the induced fields
        TM, induced_potential, induced_field = compute_product_with_polarization_matrix(
            solution, n_charges,
            pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
            pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
            pol_interaction_tensor_sr, direct_field_tensor_excl,
            eta, inverse_polarizabilities, pol_group_indices_a,
            pol_group_segment_indices, pol_group_lengths_g,
            long_range_potential_function
        )
        ene_pol = torch.dot(solution, (0.5 * TM - b_vector))

        return ene_pol, solution, induced_potential, induced_field