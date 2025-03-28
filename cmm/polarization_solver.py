import torch
import torchopt.diff.implicit
from torch_scatter import segment_csr

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
    
    def stationary_condition(self, induced_multipoles, b_vector, n_charges, 
                             pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                             pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                             pol_interaction_tensor_sr, direct_field_tensor_excl,
                             eta, inverse_polarizabilities, pol_group_indices_a,
                             pol_group_segment_indices, pol_group_lengths_g,
                             long_range_potential_function=None):
        """
        The stationary condition that must be satisfied at the solution.
        This is the residual function F(x, θ) that should be zero at the solution.
        
        Returns:
            Residual vector that should be zero at the solution
        """
        TM, _, _ = self.compute_product_with_polarization_matrix(
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
        
        # The residual is TM - b, which should be zero at the solution
        return TM - b_vector
    
    def compute_product_with_polarization_matrix(self, vec_in, n_charges,
                                                pairs_lr_i_a, pairs_lr_j_a,
                                                pairs_sr_i_a, pairs_sr_j_a,
                                                pairs_excl_i_a, pairs_excl_j_a,
                                                direct_field_tensor_lr,
                                                pol_interaction_tensor_sr,
                                                direct_field_tensor_excl,
                                                eta, alpha_inv,
                                                pol_group_indices_a,
                                                pol_group_segment_indices,
                                                pol_group_lengths_g,
                                                long_range_potential_function=None):
        
        induced_charges = vec_in[:n_charges]
        induced_dipoles = vec_in[n_charges:(4 * n_charges)].view(-1, 3)
        lagrange_muls = vec_in[(4 * n_charges):]
        induced_multipoles_a = torch.hstack((induced_charges.unsqueeze(1), induced_dipoles))
        induced_multipoles_i_lr_p = induced_multipoles_a[pairs_lr_i_a]
        induced_multipoles_i_sr_p = induced_multipoles_a[pairs_sr_i_a]

        # Get real field data
        edata_point_pairwise = torch.bmm(direct_field_tensor_lr, induced_multipoles_i_lr_p.unsqueeze(2))
        edata_ss_pairwise = torch.bmm(pol_interaction_tensor_sr, induced_multipoles_i_sr_p.unsqueeze(2))

        # Accumulate the total potentials and fields
        induced_field_data = torch.zeros(n_charges, 4, device=induced_multipoles_a.device, dtype=induced_multipoles_a.dtype)
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
        lagrange_muls_a = torch.zeros(n_charges, device=lagrange_muls.device)
        lagrange_muls_a = lagrange_muls_a.scatter_add(0, pol_group_indices_a, expanded_lagrange_muls)

        res = torch.concat((
            eta * induced_charges + lagrange_muls_a + induced_electric_potential,
            torch.bmm(alpha_inv, induced_dipoles.unsqueeze(-1)).squeeze(-1).flatten() - induced_electric_field.flatten(),
            constraints
        ))
        return res, induced_electric_potential, induced_electric_field
    
    def polarization_solve_cg(self, initial_guess, b_vector, n_charges,
                           pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
                           pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
                           pol_interaction_tensor_sr, direct_field_tensor_excl,
                           eta, inverse_polarizabilities, pol_group_indices_a,
                           pol_group_segment_indices, pol_group_lengths_g,
                           long_range_potential_function=None):
        # Precondition using whatever was put in the guess_vector
        TM0, _, _ = self.compute_product_with_polarization_matrix(
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
            TP, _, _ = self.compute_product_with_polarization_matrix(
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
            Ax, _, _ = self.compute_product_with_polarization_matrix(
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
    
    @torchopt.diff.implicit.custom_root(
        optimality_fn=lambda induced_multipoles, b_vector, n_charges, *args: 
            PolarizationSolver().stationary_condition(induced_multipoles, b_vector, n_charges, *args),
        argnums=1,
        has_aux=True
    )
    def solve(self, b_vector, n_charges, 
             pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
             pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
             pol_interaction_tensor_sr, direct_field_tensor_excl,
             eta, inverse_polarizabilities, pol_group_indices_a,
             pol_group_segment_indices, pol_group_lengths_g,
             polarizabilities=None, elec_field=None, 
             long_range_potential_function=None):
        """
        Solve the polarization equations with automatic differentiation
        handled by TorchOpt.
        """
        # Generate initial guess if we don't have a previous solution
        if self.last_induced_multipoles is None and polarizabilities is not None and elec_field is not None:
            n_groups = len(pol_group_lengths_g)
            initial_guess = self.generate_initial_guess(n_charges, n_groups, polarizabilities, elec_field)
        else:
            # Use previous solution as initial guess
            initial_guess = self.last_induced_multipoles if self.last_induced_multipoles is not None else torch.zeros_like(b_vector)
        
        # Solve the system using the selected method
        with torch.no_grad():
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
        self.last_induced_multipoles = solution.detach().clone()
        
        # Compute the induced fields
        _, induced_potential, induced_field = self.compute_product_with_polarization_matrix(
            solution, n_charges,
            pairs_lr_i_a, pairs_lr_j_a, pairs_sr_i_a, pairs_sr_j_a,
            pairs_excl_i_a, pairs_excl_j_a, direct_field_tensor_lr,
            pol_interaction_tensor_sr, direct_field_tensor_excl,
            eta, inverse_polarizabilities, pol_group_indices_a,
            pol_group_segment_indices, pol_group_lengths_g,
            long_range_potential_function
        )
        
        return (solution, (induced_potential, induced_field))
    
    def compute_polarization_energy(self, induced_multipoles, b_vector):
        """
        Compute the polarization energy from the induced multipoles.
        
        E_pol = 0.5 * x^T * (A*x - 2*b)
        where x is the induced multipoles, A is the system matrix, and b is the right-hand side.
        
        Args:
            induced_multipoles: Solution vector
            b_vector: Right-hand side vector
            
        Returns:
            Polarization energy
        """
        return 0.5 * torch.dot(induced_multipoles, -b_vector)