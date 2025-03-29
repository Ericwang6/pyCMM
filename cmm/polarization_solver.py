import torch
from .polarization import compute_product_with_polarization_matrix

class PolarizationSolver:
    """
    A polarization solver that uses the implicit function theorem via TorchOpt
    to efficiently compute induced multipoles and their derivatives.
    """
    def __init__(self, max_iter=400, tol=1e-10, solver_type="conjugate_gradient"):
        """
        Initialize the PolarizationSolver.
        
        Args:
            max_iter: Maximum number of iterations for the solver
            tol: Convergence tolerance
            solver_type: Either "conjugate_gradient" or "neumann"
        """
        self.max_iter = max_iter
        self.tol = tol
        self.solver_type = solver_type
        self.last_induced_multipoles = None
    
    def stationary_condition(self,
        induced_multipoles: torch.Tensor, b_vector: torch.Tensor,
        *args
    ):
        """
        The stationary condition that must be satisfied at the solution.
        This is the residual function F(x, θ) that should be zero at the solution.

        Returns:
            Residual vector that should be zero at the solution
        """

        n_charges, pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a, pairs_excl_i_a, pairs_excl_j_a, \
        direct_field_tensor_lr, pol_interaction_tensor_sr, direct_field_tensor_excl, \
        eta, inverse_polarizabilities, pol_group_indices_a, pol_group_segment_indices, \
        pol_group_lengths_g, long_range_potential_function = args
        TM, _, _ = compute_product_with_polarization_matrix(
            induced_multipoles, n_charges,
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

        return TM - b_vector

    def polarization_solve_cg(self, initial_guess, b_vector, n_charges,
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
        guess_vector = initial_guess.clone()
        
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
            guess_vector = guess_vector + gamma * P
            beta = 1.0 / torch.dot(residual, residual)
            residual = residual - gamma * TP
            if torch.norm(residual) < self.tol:
                return guess_vector
            beta = beta * torch.dot(residual, residual)
            P = residual + beta * P
        
        return guess_vector
    
    def polarization_solve_neumann(self, initial_guess, b_vector, n_charges, 
                                  pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                                  pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                                  pol_interaction_tensor_sr, direct_field_tensor_excl,
                                  eta, inverse_polarizabilities, pol_group_indices_a,
                                  pol_group_segment_indices, pol_group_lengths_g,
                                  long_range_potential_function=None):
        """
        Solve the polarization equations using Neumann iteration.
        
        This implements the iteration: x_{i+1} = b + (I - A)x_i
        where A is the polarization matrix and b is the right-hand side.
        """
        # For polarization systems, we can use diagonal preconditioner as the initial guess
        x = initial_guess.clone()
        
        # Damping factor
        damping = 0.7
        
        for i in range(self.max_iter):
            # Compute residual: r = b - Ax
            Ax, _, _ = compute_product_with_polarization_matrix(
                x, n_charges,
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
            residual = b_vector - Ax
            
            # Check convergence
            if torch.norm(residual) < self.tol:
                break
                
            # Update: x_{i+1} = x_i + damping * residual
            x = x + damping * residual
        
        return x
    
    def generate_initial_guess(self, n_charges, n_groups, polarizabilities, elec_field):
        """
        Generate an initial guess for the induced multipoles based on the direct field.
        """
        # Calculate direct field induced dipoles (diagonal approximation)
        dipole_part = torch.bmm(polarizabilities, elec_field.unsqueeze(-1)).squeeze(-1).flatten()
        
        # Concatenate charges (0), dipoles, and Lagrange multipliers (0)
        return torch.cat([
            torch.zeros(n_charges, device=elec_field.device),
            dipole_part,
            torch.zeros(n_groups, device=elec_field.device)
        ])
    
    def solve(self, elec_potential, elec_field, dq_groups, n_charges,
            pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
            pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
            pol_interaction_tensor_sr, direct_field_tensor_excl,
            eta, inverse_polarizabilities, pol_group_indices_a,
            pol_group_segment_indices, pol_group_lengths_g,
            polarizabilities=None,
            long_range_potential_function=None):
        class ImplicitFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, elec_potential, elec_field, dq_groups, solution, n_charges, solver, 
                        pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                        pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                        pol_interaction_tensor_sr, direct_field_tensor_excl,
                        eta, inverse_polarizabilities, pol_group_indices_a,
                        pol_group_segment_indices, pol_group_lengths_g,
                        long_range_potential_function):
                b_vector = torch.hstack((-elec_potential, elec_field.flatten(), dq_groups))
                ctx.solver = solver
                ctx.save_for_backward(elec_potential, elec_field, dq_groups, solution, n_charges,
                                      pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                                      pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                                      pol_interaction_tensor_sr, direct_field_tensor_excl,
                                      eta, inverse_polarizabilities, pol_group_indices_a,
                                      pol_group_segment_indices, pol_group_lengths_g)
                ctx.long_range_potential_function = long_range_potential_function
                return solution
            
            @staticmethod
            def backward(ctx, grad_output):
                elec_potential, elec_field, dq_groups, solution, n_charges, pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a, \
                pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr, \
                pol_interaction_tensor_sr, direct_field_tensor_excl, \
                eta, inverse_polarizabilities, pol_group_indices_a, \
                pol_group_segment_indices, pol_group_lengths_g = ctx.saved_tensors

                b_vector = torch.hstack((-elec_potential, elec_field.flatten(), dq_groups))

                solver = ctx.solver
                long_range_potential_function = ctx.long_range_potential_function

                # Define function to compute J_x^T * v product (transposed Jacobian-vector product)
                def J_x_T_mv(v):
                    with torch.enable_grad():
                        solution_temp = solution.detach().requires_grad_()
                        residual = solver.stationary_condition(
                            solution_temp, b_vector, n_charges,
                            pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                            pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                            pol_interaction_tensor_sr, direct_field_tensor_excl,
                            eta, inverse_polarizabilities, pol_group_indices_a,
                            pol_group_segment_indices, pol_group_lengths_g,
                            long_range_potential_function
                        )
                        jvp = torch.autograd.grad(
                            [torch.sum(residual * v)], [solution_temp], retain_graph=True
                        )[0]
                    return jvp

                # Solve the linear system (J_x^T) * v = grad_output using conjugate gradient
                v = torch.zeros_like(grad_output)
                r = grad_output.clone()
                p = r.clone()
                rsold = torch.sum(r * r)

                max_iter = 300  # Maximum iterations for CG
                tol = 1e-10     # Convergence tolerance

                for i in range(max_iter):
                    Ap = J_x_T_mv(p)
                    alpha = rsold / (torch.sum(p * Ap))  # Add small epsilon for stability
                    v = v + alpha * p
                    r = r - alpha * Ap
                    rsnew = torch.sum(r * r)

                    if torch.sqrt(rsnew) < tol:
                        break

                    p = r + (rsnew / rsold) * p
                    rsold = rsnew

                # Compute gradient w.r.t b_vector using vjp
                with torch.enable_grad():
                    # For elec_potential (note the negative sign in b_vector)
                    elec_potential_temp = elec_potential.detach().requires_grad_()
                    b_vector_temp = torch.hstack((-elec_potential_temp, elec_field.flatten(), dq_groups))
                    residual = solver.stationary_condition(
                        solution, b_vector_temp, n_charges,
                        pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                        pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                        pol_interaction_tensor_sr, direct_field_tensor_excl,
                        eta, inverse_polarizabilities, pol_group_indices_a,
                        pol_group_segment_indices, pol_group_lengths_g,
                        long_range_potential_function
                    )
                    grad_elec_potential = -torch.autograd.grad(
                        residual, elec_potential_temp, v, retain_graph=False
                    )[0]

                    # For elec_field
                    elec_field_temp = elec_field.detach().requires_grad_()
                    b_vector_temp = torch.hstack((-elec_potential, elec_field_temp.flatten(), dq_groups))
                    residual = solver.stationary_condition(
                        solution, b_vector_temp, n_charges,
                        pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                        pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                        pol_interaction_tensor_sr, direct_field_tensor_excl,
                        eta, inverse_polarizabilities, pol_group_indices_a,
                        pol_group_segment_indices, pol_group_lengths_g,
                        long_range_potential_function
                    )
                    grad_elec_field_flat = -torch.autograd.grad(
                        residual, elec_field_temp, v, retain_graph=False
                    )[0]
                    # Reshape to match elec_field's shape

                    # For dq_groups
                    dq_groups_temp = dq_groups.detach().requires_grad_()
                    b_vector_temp = torch.hstack((-elec_potential, elec_field.flatten(), dq_groups_temp))
                    residual = solver.stationary_condition(
                        solution, b_vector_temp, n_charges,
                        pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                        pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                        pol_interaction_tensor_sr, direct_field_tensor_excl,
                        eta, inverse_polarizabilities, pol_group_indices_a,
                        pol_group_segment_indices, pol_group_lengths_g,
                        long_range_potential_function
                    )
                    grad_dq_groups = -torch.autograd.grad(
                        residual, dq_groups_temp, v, retain_graph=False
                    )[0]
                
                return (grad_elec_potential, grad_elec_field_flat, grad_dq_groups,
                        None, None, None, None,
                        None, None, None, None,
                        None, None, None, None,
                        None, None, None, None,
                        None, None)

        # Generate initial guess if we don't have a previous solution
        if self.last_induced_multipoles is None and polarizabilities is not None and elec_field is not None:
            n_groups = len(pol_group_lengths_g)
            initial_guess = self.generate_initial_guess(n_charges, n_groups, polarizabilities, elec_field)
        else:
            # Use previous solution as initial guess
            initial_guess = self.last_induced_multipoles if self.last_induced_multipoles is not None else torch.zeros_like(b_vector)
        
        # Solve the system using the selected method
        with torch.no_grad():
            b_vector = torch.hstack((-elec_potential, elec_field.flatten(), dq_groups))
            if self.solver_type == "conjugate_gradient":
                solution = self.polarization_solve_cg(
                    initial_guess, b_vector, n_charges,
                    pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                    pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                    pol_interaction_tensor_sr, direct_field_tensor_excl,
                    eta, inverse_polarizabilities, pol_group_indices_a,
                    pol_group_segment_indices, pol_group_lengths_g,
                    long_range_potential_function
                )
            else:  # neumann
                solution = self.polarization_solve_neumann(
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
        
        # Apply the implicit function for gradients
        differentiable_solution = ImplicitFunction.apply(
            elec_potential, elec_field, dq_groups, solution, n_charges, self,
            pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
            pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
            pol_interaction_tensor_sr, direct_field_tensor_excl,
            eta, inverse_polarizabilities, pol_group_indices_a,
            pol_group_segment_indices, pol_group_lengths_g,
            long_range_potential_function
        )

        # Compute the induced fields
        TM, induced_potential, induced_field = compute_product_with_polarization_matrix(
            differentiable_solution, n_charges,
            pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
            pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
            pol_interaction_tensor_sr, direct_field_tensor_excl,
            eta, inverse_polarizabilities, pol_group_indices_a,
            pol_group_segment_indices, pol_group_lengths_g,
            long_range_potential_function
        )

        return differentiable_solution, TM, induced_potential, induced_field