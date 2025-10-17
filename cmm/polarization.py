import time

import torch
import torch.nn as nn
from torch_scatter import segment_csr
from .ewald import Ewald

import time, os
from contextlib import contextmanager

PROFILE = int(os.environ.get('CMM_PROFILE_POL', 0))

if PROFILE:
    @contextmanager
    def timer(name: str = ''):
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        if name:
            print(f"[{name}] elapsed: {elapsed:.6f} ms")
        else:
            print(f"Elapsed: {elapsed:.6f} ms")
else:
    @contextmanager
    def timer(name: str = ''):
        yield


@torch.compile
def get_field_dependent_polarizabilities(
        polarizabilities_a: torch.Tensor,
        elec_field_a: torch.Tensor,
        alpha_damp_exponent_a: torch.Tensor,
        alpha_damp_max_a: torch.Tensor
    ):
    elec_field_mag_sq_a = torch.sum(elec_field_a * elec_field_a, dim=1)
    damp_factor_a = alpha_damp_max_a * (1 - torch.exp(-alpha_damp_exponent_a * elec_field_mag_sq_a))
    return polarizabilities_a - damp_factor_a.view(-1, 1, 1) * polarizabilities_a


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
        rtol=1e-5, atol=0, maxiter=400, n_extrapolate_from=5, verbose=False,
        use_lr=True,
        use_customized_ops=True
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

        self.guess_solution = torch.tensor([], device=self.pol_group_indices_a.device)
        self.guess_error = torch.tensor([], device=self.pol_group_indices_a.device)

        self.use_customized_ops = use_customized_ops
        self.ewald = None
        self._set_ewald = False

    def set_ewald(self, alpha_ewald, k_max, device, dtype):
        self.ewald = Ewald(alpha_ewald, k_max, 1, self.use_customized_ops)
        self.ewald.to(device=device, dtype=dtype)
        self._set_ewald = True
    
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
    
    def get_extrapolated_guess_from_outputs(self):
        if self.n_solves < self.n_extrapolate_from:
            return

        m = self.n_extrapolate_from - 1
        B = self.output_cache[:m]      # (m, d)
        y = self.output_cache[-1]       # (d,)

        G = B @ B.T                    # (m, m) Gram Matrix
        t = B @ y                      # (m,)

        c = torch.linalg.solve(G, t)   # (m,)
        self.guess_solution = self.output_cache[1:].T @ c  # (d,)

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
        
        with timer("  ***POL-EXTRAPOLATE"):
            self.get_extrapolated_guess_from_outputs()
            guess_solution = self.guess_solution if self.guess_solution.numel() > 0 else guess

        with timer("  ***POL-SOLVE"):
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
            # print(info)

        with timer("  ***POL-STORE"):
            self.store_output(induced_multipoles)

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
        with timer("  POL-SCF-PREP"):
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
            Z_k = self.direct_polarization_guess_with_charge(R_k, polarizabilities, eta)

            P_k = torch.zeros_like(Z_k)

            P_k1 = P_k
            R_k1 = R_k
            R_k2 = R_k
            X_k1 = X0
            Z_k1 = Z_k
            Z_k2 = Z_k

            B_norm = torch.norm(b_vector)
            stopping_matrix = torch.max(self.rtol*B_norm, self.atol*torch.ones_like(B_norm))

        # if self.verbose:
        #     print("%03s | %010s %06s" % ("it", "dist", "it/s"))

        optimal = 0

        # time.perf_counter() is not supported by the torch.jit.script
        # start = time.perf_counter()
        # We need this extra counter variable instead of using k because of the jit
        niter = 1
        for k in range(1, self.maxiter + 1):
            # start_iter = time.perf_counter()
            # Z_k = direct_polarization_guess(R_k, self.natoms, self.n_pol_groups, polarizabilities)
            with timer("  POL-GUESS"):
                Z_k = self.direct_polarization_guess_with_charge(R_k, polarizabilities, eta)
            
            with timer("  POL-SCF-ITER-1"):
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

            with timer("  POL-SCF-ITER-2"):
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
    
    # @torch.compile
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
        with timer('  POL-PREP'):
            induced_charges = torch.narrow(vec_in, 0, 0, self.natoms)
            induced_dipoles = torch.narrow(vec_in, 0, self.natoms, 3 * self.natoms).reshape(self.natoms, 3)
            lagrange_muls = torch.narrow(vec_in, 0, self.natoms + 3 * self.natoms, vec_in.size(0) - self.natoms - 3 * self.natoms)
        with timer('  POL-REAL'):
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
            
            # pairs_i = torch.cat((pairs_lr_i_a, pairs_sr_i_a, pairs_excl_i_a))
            # pairs_j = torch.cat((pairs_lr_j_a, pairs_sr_j_a, pairs_excl_j_a))
            # tensors = torch.vstack((direct_field_tensor_lr, pol_interaction_tensor_sr, direct_field_tensor_excl))
            # edata_pairwise = torch.bmm(tensors, induced_multipoles_a[pairs_i].unsqueeze(2))
            # induced_field_data = torch.zeros((self.natoms, 4), device=induced_multipoles_a.device, dtype=induced_multipoles_a.dtype)
            # induced_field_data.scatter_add_(0, pairs_j.unsqueeze(1).expand(-1, 4), edata_pairwise.squeeze(2))
            
            # induced_field_data .mul_(torch.tensor([1, -1, -1, -1], device=pairs_lr_i_a.device).reshape(1, -1))
            induced_electric_potential = induced_field_data[:, 0]
            induced_electric_field = -induced_field_data[:, 1:4]
        
        with timer("  POL-RECIP"):
            # Get reciprocal space field data (ewald + self contribution)
            if self.use_lr:
                ewald_potential, ewald_field = self.ewald(
                    coords, box, induced_charges, induced_dipoles
                )
                induced_electric_potential = induced_electric_potential + ewald_potential
                induced_electric_field = induced_electric_field + ewald_field
        
        with timer("  POL-OTHER"):
            # Get sum of induced charges in every polarization group
            constraints = segment_csr(induced_charges[self.pol_group_indices_a], self.pol_group_segment_indices, reduce='sum')
            # constraints = torch._segment_reduce(induced_charges[self.pol_group_indices_a], 'sum', offsets=self.pol_group_segment_indices)

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
    
    def direct_polarization_guess_with_charge(self, vec_in: torch.Tensor, polarizabilities: torch.Tensor, eta: torch.Tensor):
        mean_eta = segment_csr(eta[self.pol_group_indices_a], self.pol_group_segment_indices, reduce='mean') # size n_groups
        sum_pot = segment_csr(vec_in[:self.natoms][self.pol_group_indices_a], self.pol_group_segment_indices, reduce='sum')
        multiplier = (sum_pot - mean_eta * vec_in[-self.n_pol_groups:]) / self.pol_group_lengths_g
        charges = (vec_in[:self.natoms] - multiplier.repeat_interleave(self.pol_group_lengths_g)) / mean_eta.repeat_interleave(self.pol_group_lengths_g)
        elec_field = torch.narrow(vec_in, 0, self.natoms, 3 * self.natoms).reshape(self.natoms, 3)
        dipos = torch.bmm(polarizabilities, elec_field.unsqueeze(-1)).squeeze(-1).flatten()
        return torch.cat([charges, dipos, multiplier])
    
    def direct_polarization_guess_without_charge(self, polarizabilities: torch.Tensor, elec_field: torch.Tensor):
        dipole_part = torch.bmm(polarizabilities, elec_field.unsqueeze(-1)).squeeze(-1).flatten()

        return torch.cat([
            torch.zeros(self.natoms, dtype=elec_field.dtype, device=elec_field.device),
            dipole_part,
            torch.zeros(self.n_pol_groups, dtype=elec_field.dtype, device=elec_field.device)
        ])