import os, time
from typing import Tuple
import torch
import torch.nn as nn
from torch_scatter import segment_csr
from .ewald import Ewald
from.pme_helper import PME

try:
    import torchff
    import torchff_cmm
except:
    pass

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
        rcut_sr, rcut_lr,
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
        self.pme = None
        self._set_ewald = False
        self.set_pme = False
        self.use_pme = False

        self.rcut_lr = rcut_lr
        self.rcut_sr = rcut_sr

    def set_ewald(self, alpha_ewald, k_max, device, dtype):
        self.alpha_ewald = alpha_ewald
        self.ewald = Ewald(alpha_ewald, k_max, 1, self.use_customized_ops)
        self.ewald.to(device=device, dtype=dtype)
        self._set_ewald = True
        self.use_pme = False
    def set_pme(self, alpha_ewald, k_max, device, dtype):
        self.alpha_ewald = alpha_ewald
        self.pme = PME(alpha_ewald, k_max, 1, self.use_customized_ops)
        self.pme.to(device=device, dtype=dtype)
        self._set_ewald = True
        self.use_pme = True
    
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
        eta,
        polarizabilities,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with timer("  ***POL-INV-ALPHA"):
            inverse_polarizabilities = torch.inverse(polarizabilities)
        
        with torch.no_grad():
            with timer("  ***POL-EXTRAPOLATE"):
                self.get_extrapolated_guess_from_outputs()
                guess_solution = self.guess_solution if self.guess_solution.numel() > 0 else guess

            with timer("  ***POL-SOLVE"):
                induced_multipoles, _ = self.solve(
                    coords, box, b_vector, guess_solution, 
                    eta, polarizabilities, inverse_polarizabilities, **kwargs
                )

            with timer("  ***POL-STORE"):
                self.store_output(induced_multipoles)
            
        with timer("  ***POL-ENERGY"):
            ene, ewald_field_grad = self.compute_polarization_energy(coords, box, induced_multipoles, b_vector, eta, inverse_polarizabilities, **kwargs)
        if self.use_customized_ops and ewald_field_grad is not None:
            return ene, induced_multipoles, ewald_field_grad
        else:
            return ene, induced_multipoles

    def solve(
        self,
        coords, box, b, x0, eta, polarizabilities,
        inverse_polarizabilities=None,
        **kwargs
    ):
        with timer("  POL-SCF-INIT"):
            x = x0
            r = b - self.compute_product_with_polarization_matrix(coords, box, x, eta, inverse_polarizabilities, **kwargs)
            z = self.direct_polarization_guess_with_charge(r, polarizabilities, eta)
            p = z.clone()
            rz = torch.dot(r, z)

            b_norm = torch.norm(b)
            tol = torch.max(self.rtol*b_norm, self.atol*torch.ones_like(b_norm))
        
        with timer("  POL-SCF-ITER"):
            converged = False
            for niter in range(1, self.maxiter+1):
                Ap = self.compute_product_with_polarization_matrix(coords, box, p, eta, inverse_polarizabilities, **kwargs)
                a = rz / torch.dot(p, Ap)
                x = x + a * p
                r = r - a * Ap

                if (torch.max(torch.abs(r)) <= tol).all():
                    converged = True
                    break

                z = self.direct_polarization_guess_with_charge(r, polarizabilities, eta)
                rz_new = torch.dot(r, z)
                p = z + rz_new / rz * p
                rz = rz_new

        if self.verbose:
            suffix = 'success' if converged else 'reached maxiter'
            print(f"Terminated in {niter} steps ({suffix}).")

        info = {
            "niter": niter,
            "converged": converged
        }

        return x, info
    
    def compute_product_with_polarization_matrix(
        self, 
        coords, box, vec_in, eta, inverse_polarizabilities, 
        **kwargs
    ):
        vec_out = torch.zeros(self.natoms*4+self.n_pol_groups, device=coords.device, dtype=coords.dtype)
        with timer('  POL-MATMUL-REAL'):
            induced_charges = torch.narrow(vec_in, 0, 0, self.natoms)
            induced_dipoles = torch.narrow(vec_in, 0, self.natoms, 3 * self.natoms).reshape(self.natoms, 3)
            if not self.use_customized_ops:
                induced_multipoles_a = torch.cat([induced_charges.unsqueeze(1), induced_dipoles], dim=1)

                induced_multipoles_i_lr_p = induced_multipoles_a[kwargs['pairs_lr_i_a']]
                induced_multipoles_i_sr_p = induced_multipoles_a[kwargs['pairs_sr_i_a']]

                # Get real field data
                edata_point_pairwise = torch.bmm(kwargs['direct_field_tensor_lr'], induced_multipoles_i_lr_p.unsqueeze(2))
                edata_ss_pairwise = torch.bmm(kwargs['pol_interaction_tensor_sr'], induced_multipoles_i_sr_p.unsqueeze(2))

                # Accumulate the total potentials and fields
                induced_field_data = torch.zeros((self.natoms, 4), device=induced_multipoles_a.device, dtype=induced_multipoles_a.dtype)
                induced_field_data.scatter_add_(0, kwargs['pairs_lr_j_a'].unsqueeze(1).expand(-1, 4), edata_point_pairwise.squeeze(2))
                induced_field_data.scatter_add_(0, kwargs['pairs_sr_j_a'].unsqueeze(1).expand(-1, 4), edata_ss_pairwise.squeeze(2))
                
                if self.use_lr:
                    induced_multipoles_i_excl_p = induced_multipoles_a[kwargs['pairs_excl_i_a']]
                    edata_point_excl_pairwise = torch.bmm(kwargs['direct_field_tensor_excl'], induced_multipoles_i_excl_p.unsqueeze(2))
                    induced_field_data.scatter_add_(0, kwargs['pairs_excl_j_a'].unsqueeze(1).expand(-1, 4), edata_point_excl_pairwise.squeeze(2))
                
                induced_electric_potential = induced_field_data[:, 0]
                induced_electric_field = -induced_field_data[:, 1:4]
            else:
                torch.ops.torchff.compute_cmm_polarization_real_space(
                    coords, box, kwargs['pairs'], kwargs['pairs_excl'], kwargs['b_elec_ij'], vec_in, 
                    self.alpha_ewald, self.rcut_sr, self.rcut_lr, vec_out)
        
        with timer("  POL-MATMUL-RECIP"):
            # Get reciprocal space field data (ewald + self contribution)
            if self.use_lr:
                if not self.use_customized_ops:
                    if not self.use_pme:
                        ewald_potential, ewald_field = self.ewald(
                                coords, box, induced_charges, induced_dipoles
                        )
                        induced_electric_potential = ewald_potential + induced_electric_potential
                        induced_electric_field =  ewald_field + induced_electric_field 
                    else:
                        pme_potential, pme_field = self.pme(
                                coords, box, induced_charges, induced_dipoles
                        )
                        induced_electric_potential = pme_potential + induced_electric_potential
                        induced_electric_field =  pme_field + induced_electric_field 
                else:
                    if not self.use_pme:
                        ewald_potential, ewald_field, _, _, _ = self.ewald(
                                coords, box, induced_charges, induced_dipoles
                        )
                        induced_electric_potential = ewald_potential
                        induced_electric_field = ewald_field
                    else:
                        pme_potential, pme_field, _, _, _ = self.pme(
                                coords, box, induced_charges, induced_dipoles
                        )
                        induced_electric_potential = pme_potential
                        induced_electric_field = pme_field
        
        with timer("  POL-MATMUL-CHARGE"):
            # Get sum of induced charges in every polarization group
            constraints = segment_csr(induced_charges[self.pol_group_indices_a], self.pol_group_segment_indices, reduce='sum')
            # constraints = torch._segment_reduce(induced_charges[self.pol_group_indices_a], 'sum', offsets=self.pol_group_segment_indices)

            # Expand lagrange multipliers from group index space to atomic index space
            expanded_lagrange_muls = vec_in[-self.n_pol_groups:].repeat_interleave(self.pol_group_lengths_g)

            # Scatter these values back to the atomic indices
            lagrange_muls_a = torch.zeros(self.natoms, device=coords.device)
            lagrange_muls_a.scatter_add_(0, self.pol_group_indices_a, expanded_lagrange_muls)
        
        with timer("  POL-MATMUL-OTHER"):
            vec_out[:self.natoms] += eta * induced_charges + lagrange_muls_a + induced_electric_potential
            vec_out[self.natoms:self.natoms*4] += torch.bmm(inverse_polarizabilities, induced_dipoles.unsqueeze(-1)).squeeze(-1).flatten() - induced_electric_field.flatten()
            vec_out[-self.n_pol_groups:] += constraints
        return vec_out
    
    def compute_polarization_energy(
        self, coords, box, induced_multipoles, b_vector, eta, inverse_polarizabilities, **kwargs
    ):
        if not self.use_customized_ops:
            tmp = self.compute_product_with_polarization_matrix(coords, box, induced_multipoles, eta, inverse_polarizabilities, **kwargs)
            return torch.dot(induced_multipoles, 0.5*tmp-b_vector), None
        else:
            vec_out = torch.zeros(self.natoms*4+self.n_pol_groups, device=coords.device, dtype=coords.dtype)
            with timer("  POL-MATMUL-RECIP"):
                # Get reciprocal space field data (ewald + self contribution)
                if self.use_lr:
                    induced_charges = torch.narrow(induced_multipoles, 0, 0, self.natoms)
                    induced_dipoles = torch.narrow(induced_multipoles, 0, self.natoms, 3 * self.natoms).reshape(self.natoms, 3) 
                    if self.use_pme:
                        ewald_potential, ewald_field, ewald_field_grad, _, _ = self.pme(
                                coords, box, induced_charges, induced_dipoles
                        )
                    else:
                        ewald_potential, ewald_field, ewald_field_grad, _, _ = self.ewald(
                                coords, box, induced_charges, induced_dipoles
                        )

            with timer("  POL-MATMUL-CHARGE"):
                # Get sum of induced charges in every polarization group
                constraints = segment_csr(induced_charges[self.pol_group_indices_a], self.pol_group_segment_indices, reduce='sum')
                # constraints = torch._segment_reduce(induced_charges[self.pol_group_indices_a], 'sum', offsets=self.pol_group_segment_indices)

                # Expand lagrange multipliers from group index space to atomic index space
                expanded_lagrange_muls = induced_multipoles[-self.n_pol_groups:].repeat_interleave(self.pol_group_lengths_g)

                # Scatter these values back to the atomic indices
                lagrange_muls_a = torch.zeros(self.natoms, device=coords.device)
                lagrange_muls_a.scatter_add_(0, self.pol_group_indices_a, expanded_lagrange_muls)
        
            with timer("  POL-MATMUL-OTHER"):
                vec_out[:self.natoms] += eta * induced_charges + lagrange_muls_a + ewald_potential
                vec_out[self.natoms:self.natoms*4] += torch.bmm(inverse_polarizabilities, induced_dipoles.unsqueeze(-1)).squeeze(-1).flatten() - ewald_field.flatten()
                vec_out[-self.n_pol_groups:] += constraints

            ene = torch.dot(induced_multipoles, 0.5*vec_out-b_vector) + torch.ops.torchff.cmm_polarization_energy_from_induced_multipoles(
                kwargs['dist_vecs'], kwargs['pairs'], kwargs['dist_vecs_excl'], kwargs['pairs_excl'],
                induced_multipoles, kwargs['b_elec_ij'], self.alpha_ewald, self.rcut_sr, self.rcut_lr, self.natoms
            )
            return ene, ewald_field_grad
        
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
