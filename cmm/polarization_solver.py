import torch
from .polarization import compute_product_with_polarization_matrix

import torch
import time


def cg_solve(A_mm, b, M_mm=None, X0=None, rtol=1e-7, atol=1e-7, maxiter=400, verbose=False):
    """Solves positive-definite matrix linear system using the preconditioned CG algorithm.
    This implementation is a modified version of that available at: https://github.com/sbarratt/torch_cg/
    which is MIT licensed.

    This function solves a linear system of the form

        A x = b

    where A is a n x n positive definite matrix and b is a n element vector,
    and x is the n element vector representing the solution.

    Args:
        A_bmm: A callable that performs a matrix multiply of A and X.
        b: A vector representing the right hand side.
        M_mm: (optional) A callable that performs a matrix multiply of the preconditioning
            matrix M and an n x n matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """

    if M_mm is None:
        M_mm = lambda x: x
        X0 = torch.zeros_like(b)
    if X0 is None:
        X0 = M_mm(b)

    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = b - A_mm(X_k)
    Z_k = M_mm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(b)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_mm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = torch.dot(R_k2, Z_k2)
            #denominator[denominator == 0] = 1e-8
            beta = torch.dot(R_k1, Z_k1) / denominator
            P_k = Z_k1 + beta * P_k1

        AP_k = A_mm(P_k)
        denominator = torch.dot(P_k, AP_k)
        #denominator[denominator == 0] = 1e-8
        alpha = torch.dot(R_k1, Z_k1) / denominator
        X_k = X_k1 + alpha * P_k
        R_k = R_k1 - alpha * AP_k
        end_iter = time.perf_counter()

        residual_norm = torch.norm(R_k)

        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, torch.max(residual_norm-stopping_matrix),
                    1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))

    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info


class CG(torch.autograd.Function):

    def __init__(self, A_mm, M_mm=None, rtol=1e-7, atol=1e-7, maxiter=400, verbose=False):
        self.A_bmm = A_mm
        self.M_bmm = M_mm
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.verbose = verbose

    @staticmethod
    def forward(self, B, X0=None):
        X, _ = cg_solve(self.A_mm, B, M_mm=self.M_bmm, X0=X0, rtol=self.rtol,
                     atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return X

    @staticmethod
    def backward(self, dX):
        dB, _ = cg_solve(self.A_mm, dX, M_mm=self.M_mm, rtol=self.rtol,
                      atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return dB

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