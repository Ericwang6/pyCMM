from typing import Tuple, Optional
import time

import torch
import torch.nn as nn
from torch_scatter import segment_csr
from .ewald import long_range_potential_rank_1
from .polarization import direct_polarization_guess

def cg_solve(A_mm, b, M_mm=None, X0=None, rtol=1e-7, atol=0, maxiter=400, verbose=False):
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
            beta = torch.dot(R_k1, Z_k1) / denominator
            P_k = Z_k1 + beta * P_k1

        AP_k = A_mm(P_k)
        denominator = torch.dot(P_k, AP_k)
        alpha = torch.dot(R_k1, Z_k1) / denominator
        X_k = X_k1 + alpha * P_k
        R_k = R_k1 - alpha * AP_k
        end_iter = time.perf_counter()

        #residual_norm = torch.norm(R_k)
        residual_norm = torch.max(torch.abs(R_k))

        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, torch.max(residual_norm-stopping_matrix),
                    1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if not optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (success). Took %.3f ms." %
                  (k, (end - start) * 1000))

    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info


class CG(torch.autograd.Function):
    # TODO: This doesn't actually work properly when going through the apply method
    # since we need to store some tensors or something. I still don't really understand
    # where the ctx variable comes from. Try doing this with backward hooks.
    def __init__(self, A_mm, M_mm=None, rtol=1e-7, atol=0, maxiter=400, n_extrapolate_from=10, verbose=False):
        self.A_mm = A_mm
        self.M_mm = M_mm
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.verbose = verbose
        self.info_forward = None
        self.info_backward = None

        self.n_extrapolate_from = n_extrapolate_from
        self.n_solves = 0
        self.n_extrapolations = 0
        self.input_cache = None
        self.output_cache = None
        self.extrapolated_input_cache = None
        self.extrapolated_output_cache = None

    def store_input(self, b_vector: torch.Tensor):
        index = self.n_solves
        if self.n_solves == 0:
            self.input_cache = torch.zeros((self.n_extrapolate_from, b_vector.size(0)), dtype=b_vector.dtype, device=b_vector.device)
        elif self.n_solves >= self.n_extrapolate_from:
            # roll everything so that the first cached entry is dropped
            # (or really wrapped around to the back to be replaced)
            self.input_cache = torch.roll(self.input_cache, shifts=-1, dims=0)
            index = self.n_extrapolate_from - 1
        
        # Store newest input-output pair in the caches #
        self.input_cache[index, :] = b_vector
    
    def store_output(self, solution_vector: torch.Tensor):
        index = self.n_solves
        if self.n_solves == 0:
            self.output_cache = torch.zeros((self.n_extrapolate_from, solution_vector.size(0)), dtype=solution_vector.dtype, device=solution_vector.device)
        elif self.n_solves >= self.n_extrapolate_from:
            # roll everything so that the first in-out pair is dropped
            # (or really wrapped around to the back to be replaced)
            self.output_cache = torch.roll(self.output_cache, shifts=-1, dims=0)
            index = self.n_extrapolate_from - 1
        
        # Store newest input-output pair in the caches #
        self.output_cache[index, :] = solution_vector
        self.n_solves += 1
    
    def store_extrapolated_output(self, guess_solution_vector: torch.Tensor):
        index = self.n_extrapolations
        if self.n_extrapolations == 0:
            self.extrapolated_output_cache = torch.zeros((self.n_extrapolate_from, guess_solution_vector.size(0)), dtype=guess_solution_vector.dtype, device=guess_solution_vector.device)
        elif self.n_extrapolations >= self.n_extrapolate_from:
            # roll everything so that the first in-out pair is dropped
            # (or really wrapped around to the back to be replaced)
            self.extrapolated_output_cache = torch.roll(self.extrapolated_output_cache, shifts=-1, dims=0)
            index = self.n_extrapolate_from - 1
        
        # Store newest input-output pair in the caches #
        self.extrapolated_output_cache[index, :] = guess_solution_vector
        self.n_extrapolations += 1

    def get_extrapolated_guess_from_inputs(self):
        # @SPEED: Can avoid recomputing the matmuls for every extrapolation.
        # Basically, we fill in the extrapolations for only the most recently updated
        # vectors once we have done one extrapolation. Need to store the data which
        # currently gets allocated each time.

        # TODO: Clean up the sizes of things. Explicitly store the most recent inputs
        # and solutions as attributes. Then once we have a new solution move the
        # previous solution into the cache. Currently we actually extrapolate with
        # one less solution than promised since we store the inputs in the cache immediately.
        if self.n_solves >= self.n_extrapolate_from:
            # Extrapolate using the full set of solutions #
            extrapolation_matrix = torch.zeros((self.n_extrapolate_from-1, self.n_extrapolate_from-1),
                                               dtype=self.input_cache[0].dtype,
                                               device=self.input_cache[0].device)
            extrapolation_target = torch.zeros(self.n_extrapolate_from-1,
                                   dtype=self.input_cache[0].dtype,
                                   device=self.input_cache[0].device)
            for i in range(self.n_extrapolate_from-1):
                extrapolation_target[i] = torch.matmul(self.input_cache[i], self.input_cache[-1])
                for j in range(i, self.n_extrapolate_from-1):
                    extrapolation_matrix[i, j] = torch.matmul(self.input_cache[i], self.input_cache[j])
                    extrapolation_matrix[j, i] = extrapolation_matrix[i, j]
            
            extrapolation_coeffs = torch.linalg.lstsq(extrapolation_matrix, extrapolation_target).solution
            # TODO: Store the extrapolated inputs and outputs for second-order correction.
            guess_input = torch.matmul(self.input_cache[:-1, :].T, extrapolation_coeffs)
            guess_solution = torch.matmul(self.output_cache[1:, :].T, extrapolation_coeffs)
            return guess_solution
        return None
    
    def get_extrapolated_guess_from_outputs(self):
        guess_solution = None
        guess_error = None
        if self.n_solves >= self.n_extrapolate_from:
            # Extrapolate using the full set of solutions #
            extrapolation_matrix = torch.zeros((self.n_extrapolate_from-1, self.n_extrapolate_from-1),
                                               dtype=self.output_cache[0].dtype,
                                               device=self.output_cache[0].device)
            extrapolation_target = torch.zeros(self.n_extrapolate_from-1,
                                   dtype=self.output_cache[0].dtype,
                                   device=self.output_cache[0].device)
            for i in range(self.n_extrapolate_from-1):
                extrapolation_target[i] = torch.matmul(self.output_cache[i], self.output_cache[-1])
                for j in range(i, self.n_extrapolate_from-1):
                    extrapolation_matrix[i, j] = torch.matmul(self.output_cache[i], self.output_cache[j])
                    extrapolation_matrix[j, i] = extrapolation_matrix[i, j]
            
            extrapolation_coeffs = torch.linalg.lstsq(extrapolation_matrix, extrapolation_target).solution
            # TODO: Store the extrapolated inputs and outputs for second-order correction.
            #guess_input = torch.matmul(self.input_cache[:-1, :].T, extrapolation_coeffs)
            guess_solution = torch.matmul(self.output_cache[1:, :].T, extrapolation_coeffs)
            
        # If we have enough extrapolation data, try a second-order extrapolation #
        if self.n_extrapolations >= self.n_extrapolate_from:
            # Extrapolate using the full set of solutions #
            extrapolation_matrix = torch.zeros((self.n_extrapolate_from-1, self.n_extrapolate_from-1),
                                               dtype=self.extrapolated_output_cache[0].dtype,
                                               device=self.extrapolated_output_cache[0].device)
            extrapolation_target = torch.zeros(self.n_extrapolate_from-1,
                                   dtype=self.extrapolated_output_cache[0].dtype,
                                   device=self.extrapolated_output_cache[0].device)
            for i in range(self.n_extrapolate_from-1):
                extrapolation_target[i] = torch.matmul(self.extrapolated_output_cache[i], self.extrapolated_output_cache[-1])
                for j in range(i, self.n_extrapolate_from-1):
                    extrapolation_matrix[i, j] = torch.matmul(self.extrapolated_output_cache[i], self.extrapolated_output_cache[j])
                    extrapolation_matrix[j, i] = extrapolation_matrix[i, j]
            
            extrapolation_coeffs = torch.linalg.lstsq(extrapolation_matrix, extrapolation_target).solution
            guess_error = torch.matmul(self.extrapolated_output_cache[1:, :].T, extrapolation_coeffs)

        # TODO: Clean up the guess solution by ensuring that we exactly respect the constraints
        return guess_solution, guess_error

    # NOTE(JOE): This method is here in case you want to solve without doing
    # the backward pass to get derivatives of the induced dipoles.
    def solve(self, B, X0=None):
        self.store_input(B)
        # TODO: Can also try an extrapolation based on input-output pairs
        #guess_solution = self.get_extrapolated_guess_from_inputs()
        # NOTE(JOE): Second-order extrapolation isn't helping at all. Not sure if I implemented it wrong or what.
        guess_solution, guess_error = self.get_extrapolated_guess_from_outputs()
        if guess_solution is None:
            guess_solution = X0
        #if guess_error is not None:
        #    guess_solution += guess_error
        X, self.info_forward = cg_solve(self.A_mm, B, M_mm=self.M_mm, X0=guess_solution, rtol=self.rtol,
            atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        #print(f"Rel. Error: {torch.norm(guess_solution - X) / torch.norm(X):.5f}, Abs. Error: {torch.max(torch.abs(guess_solution - X)):.5f}")
        self.store_output(X)
        if self.n_solves >= self.n_extrapolate_from:
            if guess_error is None:
                self.store_extrapolated_output(guess_solution - X)
            else:
                self.store_extrapolated_output(guess_solution - guess_error - X)
        #X.register_hook(lambda dX : cg_solve(self.A_mm, dX, M_mm=self.M_mm, rtol=self.rtol,
        #             atol=self.atol, maxiter=self.maxiter, verbose=self.verbose))
        return X

    @staticmethod
    def forward(self, B, X0=None):
        X, self.info_forward = cg_solve(self.A_mm, B, M_mm=self.M_mm, X0=X0, rtol=self.rtol,
                     atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return X

    @staticmethod
    def backward(self, dX):
        dB, self.info_backward = cg_solve(self.A_mm, dX, M_mm=self.M_mm, rtol=self.rtol,
                      atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return dB


class CMMPolarization(nn.Module):
    '''
    This is a re-write version of CG class to support torch.jit.script
    '''
    def __init__(
        self,
        natoms,
        pol_group_indices_a,
        pol_group_segment_indices,
        pol_group_lengths_g,
        rtol=1e-7, atol=0, maxiter=400, n_extrapolate_from=10, verbose=False,
        use_lr=True, alpha_ewald=0.0, k_max=0
    ):
        super().__init__()
        self.natoms = natoms
        self.pol_group_indices_a = pol_group_indices_a
        self.pol_group_segment_indices = pol_group_segment_indices
        self.pol_group_lengths_g = pol_group_lengths_g
        self.n_pol_groups = self.pol_group_lengths_g.size(0)

        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.verbose = verbose
        self.use_lr = use_lr

        self.n_extrapolate_from = n_extrapolate_from
        self.n_solves = 0
        self.n_extrapolations = 0
        self.input_cache = torch.tensor([], device=self.pol_group_indices_a.device)
        self.output_cache = torch.tensor([], device=self.pol_group_indices_a.device)
        self.extrapolated_input_cache = torch.tensor([], device=self.pol_group_indices_a.device)
        self.extrapolated_output_cache = torch.tensor([], device=self.pol_group_indices_a.device)

        self.alpha_ewald = alpha_ewald
        self.k_max = k_max

        self.guess_solution = torch.tensor([], device=self.pol_group_indices_a.device)
        self.guess_error = torch.tensor([], device=self.pol_group_indices_a.device)
    
    def store_input(self, b_vector: torch.Tensor):
        index = self.n_solves
        if self.n_solves == 0:
            self.input_cache = torch.zeros((self.n_extrapolate_from, b_vector.size(0)), dtype=b_vector.dtype, device=b_vector.device)
        elif self.n_solves >= self.n_extrapolate_from:
            # roll everything so that the first cached entry is dropped
            # (or really wrapped around to the back to be replaced)
            self.input_cache = torch.roll(self.input_cache, shifts=-1, dims=0)
            index = self.n_extrapolate_from - 1
        
        # Store newest input-output pair in the caches #
        self.input_cache[index, :] = b_vector
    
    def store_output(self, solution_vector: torch.Tensor):
        index = self.n_solves
        if self.n_solves == 0:
            self.output_cache = torch.zeros((self.n_extrapolate_from, solution_vector.size(0)), dtype=solution_vector.dtype, device=solution_vector.device)
        elif self.n_solves >= self.n_extrapolate_from:
            # roll everything so that the first in-out pair is dropped
            # (or really wrapped around to the back to be replaced)
            self.output_cache = torch.roll(self.output_cache, shifts=-1, dims=0)
            index = self.n_extrapolate_from - 1
        
        # Store newest input-output pair in the caches #
        self.output_cache[index, :] = solution_vector
        self.n_solves += 1
    
    def store_extrapolated_output(self, guess_solution_vector: torch.Tensor):
        index = self.n_extrapolations
        if self.n_extrapolations == 0:
            self.extrapolated_output_cache = torch.zeros((self.n_extrapolate_from, guess_solution_vector.size(0)), dtype=guess_solution_vector.dtype, device=guess_solution_vector.device)
        elif self.n_extrapolations >= self.n_extrapolate_from:
            # roll everything so that the first in-out pair is dropped
            # (or really wrapped around to the back to be replaced)
            self.extrapolated_output_cache = torch.roll(self.extrapolated_output_cache, shifts=-1, dims=0)
            index = self.n_extrapolate_from - 1
        
        # Store newest input-output pair in the caches #
        self.extrapolated_output_cache[index, :] = guess_solution_vector
        self.n_extrapolations += 1

    def get_extrapolated_guess_from_inputs(self):
        # @SPEED: Can avoid recomputing the matmuls for every extrapolation.
        # Basically, we fill in the extrapolations for only the most recently updated
        # vectors once we have done one extrapolation. Need to store the data which
        # currently gets allocated each time.

        # TODO: Clean up the sizes of things. Explicitly store the most recent inputs
        # and solutions as attributes. Then once we have a new solution move the
        # previous solution into the cache. Currently we actually extrapolate with
        # one less solution than promised since we store the inputs in the cache immediately.
        if self.n_solves >= self.n_extrapolate_from:
            # Extrapolate using the full set of solutions #
            extrapolation_matrix = torch.zeros((self.n_extrapolate_from-1, self.n_extrapolate_from-1),
                                               dtype=self.input_cache.dtype,
                                               device=self.input_cache.device)
            extrapolation_target = torch.zeros(self.n_extrapolate_from-1,
                                   dtype=self.input_cache.dtype,
                                   device=self.input_cache.device)
            for i in range(self.n_extrapolate_from-1):
                extrapolation_target[i] = torch.matmul(self.input_cache[i], self.input_cache[-1])
                for j in range(i, self.n_extrapolate_from-1):
                    extrapolation_matrix[i, j] = torch.matmul(self.input_cache[i], self.input_cache[j])
                    extrapolation_matrix[j, i] = extrapolation_matrix[i, j]
            
            extrapolation_coeffs = torch.linalg.lstsq(extrapolation_matrix, extrapolation_target).solution
            # TODO: Store the extrapolated inputs and outputs for second-order correction.
            guess_input = torch.matmul(self.input_cache[:-1, :].T, extrapolation_coeffs)
            guess_solution = torch.matmul(self.output_cache[1:, :].T, extrapolation_coeffs)
            return guess_solution
        return None
    
    def get_extrapolated_guess_from_outputs(self):
        if self.n_solves >= self.n_extrapolate_from:
            # Extrapolate using the full set of solutions #
            extrapolation_matrix = torch.zeros((self.n_extrapolate_from-1, self.n_extrapolate_from-1),
                                               dtype=self.output_cache.dtype,
                                               device=self.output_cache.device)
            extrapolation_target = torch.zeros(self.n_extrapolate_from-1,
                                   dtype=self.output_cache.dtype,
                                   device=self.output_cache.device)
            for i in range(self.n_extrapolate_from-1):
                extrapolation_target[i] = torch.matmul(self.output_cache[i], self.output_cache[-1])
                for j in range(i, self.n_extrapolate_from-1):
                    extrapolation_matrix[i, j] = torch.matmul(self.output_cache[i], self.output_cache[j])
                    extrapolation_matrix[j, i] = extrapolation_matrix[i, j]
            
            extrapolation_coeffs = torch.linalg.lstsq(extrapolation_matrix, extrapolation_target).solution
            # TODO: Store the extrapolated inputs and outputs for second-order correction.
            #guess_input = torch.matmul(self.input_cache[:-1, :].T, extrapolation_coeffs)
            self.guess_solution = torch.matmul(self.output_cache[1:, :].T, extrapolation_coeffs)
            
        # If we have enough extrapolation data, try a second-order extrapolation #
        if self.n_extrapolations >= self.n_extrapolate_from:
            # Extrapolate using the full set of solutions #
            extrapolation_matrix = torch.zeros((self.n_extrapolate_from-1, self.n_extrapolate_from-1),
                                               dtype=self.extrapolated_output_cache.dtype,
                                               device=self.extrapolated_output_cache.device)
            extrapolation_target = torch.zeros(self.n_extrapolate_from-1,
                                   dtype=self.extrapolated_output_cache.dtype,
                                   device=self.extrapolated_output_cache.device)
            for i in range(self.n_extrapolate_from-1):
                extrapolation_target[i] = torch.matmul(self.extrapolated_output_cache[i], self.extrapolated_output_cache[-1])
                for j in range(i, self.n_extrapolate_from-1):
                    extrapolation_matrix[i, j] = torch.matmul(self.extrapolated_output_cache[i], self.extrapolated_output_cache[j])
                    extrapolation_matrix[j, i] = extrapolation_matrix[i, j]
            
            extrapolation_coeffs = torch.linalg.lstsq(extrapolation_matrix, extrapolation_target).solution
            self.guess_error = torch.matmul(self.extrapolated_output_cache[1:, :].T, extrapolation_coeffs)

    def forward(
        self,
        coords,
        box,
        b_vector,
        guess,
        pairs_lr_i_a, pairs_lr_j_a,
        pairs_sr_i_a, pairs_sr_j_a,
        pairs_excl_i_a, pairs_excl_j_a,
        direct_field_tensor_lr,
        pol_interaction_tensor_sr,
        direct_field_tensor_excl,
        eta,
        alpha,
        alpha_inv
    ):
        
        #self.store_input(b_vector)
        #self.get_extrapolated_guess_from_outputs()
        guess_solution = guess

        induced_multipoles, info = self.solve(
            coords, box, b_vector, guess_solution, 
            pairs_lr_i_a, pairs_lr_j_a,
            pairs_sr_i_a, pairs_sr_j_a,
            pairs_excl_i_a, pairs_excl_j_a,
            direct_field_tensor_lr,
            pol_interaction_tensor_sr,
            direct_field_tensor_excl,
            eta,
            alpha,
            alpha_inv
        )

        #self.store_output(induced_multipoles)
        #if self.n_solves >= self.n_extrapolate_from:
        #    if self.guess_error.numel() == 0:
        #        self.store_extrapolated_output(guess_solution - induced_multipoles)
        #    else:
        #        self.store_extrapolated_output(guess_solution - self.guess_error - induced_multipoles)

        return induced_multipoles
    
    
    def solve(
        self,
        coords,
        box,
        b_vector,
        X0,
        pairs_lr_i_a, pairs_lr_j_a,
        pairs_sr_i_a, pairs_sr_j_a,
        pairs_excl_i_a, pairs_excl_j_a,
        direct_field_tensor_lr,
        pol_interaction_tensor_sr,
        direct_field_tensor_excl,
        eta,
        polarizabilities,
        inverse_polarizabilities
    ):
        
        X_k = X0
        R_k = b_vector - self.compute_product_with_polarization_matrix(
            coords, box, X_k,
            pairs_lr_i_a, pairs_lr_j_a,
            pairs_sr_i_a, pairs_sr_j_a,
            pairs_excl_i_a, pairs_excl_j_a,
            direct_field_tensor_lr,
            pol_interaction_tensor_sr,
            direct_field_tensor_excl,
            eta,
            inverse_polarizabilities
        )
        Z_k = direct_polarization_guess(R_k, self.natoms, self.n_pol_groups, polarizabilities)

        P_k = torch.zeros_like(Z_k)

        P_k1 = P_k
        R_k1 = R_k
        R_k2 = R_k
        X_k1 = X0
        Z_k1 = Z_k
        Z_k2 = Z_k

        B_norm = torch.norm(b_vector)
        stopping_matrix = torch.max(self.rtol*B_norm, self.atol*torch.ones_like(B_norm))

        if self.verbose:
            print("%03s | %010s %06s" % ("it", "dist", "it/s"))

        optimal = 0

        # time.perf_counter() is not supported by the torch.jit.script
        # start = time.perf_counter()
        # We need this extra counter variable instead of using k because of the jit
        niter = 1
        for k in range(1, self.maxiter + 1):
            # start_iter = time.perf_counter()
            Z_k = direct_polarization_guess(R_k, self.natoms, self.n_pol_groups, polarizabilities)

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
                beta = torch.dot(R_k1, Z_k1) / denominator
                P_k = Z_k1 + beta * P_k1

            AP_k = self.compute_product_with_polarization_matrix(
                coords, box, P_k,
                pairs_lr_i_a, pairs_lr_j_a,
                pairs_sr_i_a, pairs_sr_j_a,
                pairs_excl_i_a, pairs_excl_j_a,
                direct_field_tensor_lr,
                pol_interaction_tensor_sr,
                direct_field_tensor_excl,
                eta,
                inverse_polarizabilities
            )
            denominator = torch.dot(P_k, AP_k)
            alpha = torch.dot(R_k1, Z_k1) / denominator
            X_k = X_k1 + alpha * P_k
            R_k = R_k1 - alpha * AP_k
            # end_iter = time.perf_counter()

            #residual_norm = torch.norm(R_k)
            residual_norm = torch.max(torch.abs(R_k))

            # if self.verbose:
            #     print("%03d | %8.4e %4.2f" %
            #         (k, torch.max(residual_norm-stopping_matrix),
            #             1. / (end_iter - start_iter)))

            if (residual_norm <= stopping_matrix).all():
                optimal = 1
                break
            niter += 1

        # end = time.perf_counter()

        if self.verbose:
            suffix = 'success' if optimal else 'reached maxiter'
            print(f"Terminated in {niter} steps ({suffix}).")

        info = {
            "niter": niter,
            "optimal": optimal
        }

        return X_k, info


    def compute_product_with_polarization_matrix(
        self, 
        coords, box,
        vec_in,
        pairs_lr_i_a, pairs_lr_j_a,
        pairs_sr_i_a, pairs_sr_j_a,
        pairs_excl_i_a, pairs_excl_j_a,
        direct_field_tensor_lr,
        pol_interaction_tensor_sr,
        direct_field_tensor_excl,
        eta,
        inverse_polarizabilities
    ):
        induced_charges = torch.narrow(vec_in, 0, 0, self.natoms)
        induced_dipoles = torch.narrow(vec_in, 0, self.natoms, 3 * self.natoms).reshape(self.natoms, 3)
        lagrange_muls = torch.narrow(vec_in, 0, self.natoms + 3 * self.natoms, vec_in.size(0) - self.natoms - 3 * self.natoms)
        induced_multipoles_a = torch.cat([induced_charges.unsqueeze(1), induced_dipoles], dim=1)

        induced_multipoles_i_lr_p = induced_multipoles_a[pairs_lr_i_a]
        induced_multipoles_i_sr_p = induced_multipoles_a[pairs_sr_i_a]

        # Get real field data
        edata_point_pairwise = torch.bmm(direct_field_tensor_lr, induced_multipoles_i_lr_p.unsqueeze(2))
        edata_ss_pairwise = torch.bmm(pol_interaction_tensor_sr, induced_multipoles_i_sr_p.unsqueeze(2))

        # Accumulate the total potentials and fields
        induced_field_data = torch.zeros((self.natoms, 4), device=induced_multipoles_a.device, dtype=induced_multipoles_a.dtype)
        induced_field_data.scatter_add_(0, pairs_lr_j_a.unsqueeze(1).expand(-1, 4), edata_point_pairwise.squeeze(2))
        induced_field_data.scatter_add_(0, pairs_sr_j_a.unsqueeze(1).expand(-1, 4), edata_ss_pairwise.squeeze(2))
        
        if self.use_lr:
            induced_multipoles_i_excl_p = induced_multipoles_a[pairs_excl_i_a]
            edata_point_excl_pairwise = torch.bmm(direct_field_tensor_excl, induced_multipoles_i_excl_p.unsqueeze(2))
            induced_field_data.scatter_add_(0, pairs_excl_j_a.unsqueeze(1).expand(-1, 4), edata_point_excl_pairwise.squeeze(2))
        
        induced_field_data.mul_(torch.tensor([1, -1, -1, -1], device=pairs_lr_i_a.device).reshape(1, -1))
        induced_electric_potential = induced_field_data[:, 0]
        induced_electric_field = induced_field_data[:, 1:4]

        # Get reciprocal space field data (ewald + self contribution)
        if self.use_lr:
            ewald_potential, ewald_field = long_range_potential_rank_1(
                coords, induced_charges, induced_dipoles, box, self.alpha_ewald, self.k_max
            )
            induced_electric_potential = induced_electric_potential + ewald_potential
            induced_electric_field = induced_electric_field + ewald_field

        # Get sum of induced charges in every polarization group
        constraints = segment_csr(induced_charges[self.pol_group_indices_a], self.pol_group_segment_indices, reduce='sum')

        # Expand lagrange multipliers from group index space to atomic index space
        expanded_lagrange_muls = lagrange_muls.repeat_interleave(self.pol_group_lengths_g)

        # Scatter these values back to the atomic indices
        lagrange_muls_a = torch.zeros(self.natoms, device=lagrange_muls.device)
        lagrange_muls_a.scatter_add_(0, self.pol_group_indices_a, expanded_lagrange_muls)

        residual = torch.concat((
            eta * induced_charges + lagrange_muls_a + induced_electric_potential,
            torch.bmm(inverse_polarizabilities, induced_dipoles.unsqueeze(-1)).squeeze(-1).flatten() - induced_electric_field.flatten(),
            constraints
        ))
        return residual
