import torch, math
from torch_scatter import segment_csr
from .electrostatics import computeDampFactorsErfc, computeDampFactorsErf
from .ewald import long_range_potential, long_range_potential_rank_1
from .dispersion import computeDispersionFromPairs, compute_long_range_dispersion_correction
from .coordinate_manager import CoordinateManager
from .parameters import Parameterizer
from .topology import Topology
from .terms import *
from .units import *
from .switching_functions import switch_543
from .force_field import ForceField

class SPCfw(ForceField):
    def __init__(self,
                 cutoff_ewald: torch.Tensor=torch.tensor(9.0 / BOHR2ANG, dtype=torch.float64),
                 ewald_tolerance: torch.Tensor=torch.tensor(1e-6, dtype=torch.float64)) -> None:
        super().__init__()
        # These are local indices for the atom types, not the actual
        # atom type indices which are decided by the Parameterizer.
        self._types_to_index = {
            "O_water": 0, "H_water": 1
        }
        self.ewald_tolerance = ewald_tolerance if torch.is_tensor(ewald_tolerance) else torch.tensor(ewald_tolerance, dtype=torch.float64)
        self.cutoff_ewald = cutoff_ewald if torch.is_tensor(cutoff_ewald) else torch.tensor(cutoff_ewald, dtype=torch.float64)
        self._build()
    
    def _build(self):
        self.mono = torch.tensor([
            -0.82, 0.41, # Water
        ])
        self._raw_atomic_params = {
            "mono": self.mono,
        }
        self.pair_params = {
            ("O_water", "H_water"): {
                "k_b": torch.tensor([1059.162 / HARTREE2KCAL * BOHR2ANG * BOHR2ANG]),
                "r_eq": torch.tensor([1.012 / BOHR2ANG]),
            },
            ("O_water", "O_water"): {
                "eps_lj": torch.tensor([0.1554253 / HARTREE2KCAL]),
                "sigma_lj": torch.tensor([3.165492 / BOHR2ANG]),
            },
        }

        self.angle_params = {
            ("H_water", "O_water", "H_water"): {
                "theta_eq": torch.tensor([113.24 * math.pi / 180.0]),
                "k_theta": torch.tensor([75.90 / HARTREE2KCAL]),
            }
        }

        with torch.no_grad():
            self.atomic_params = {}
            for type_key in self._types_to_index.keys():
                these_atomic_params = {}
                for param_key in self._raw_atomic_params.keys():
                    these_atomic_params[param_key] = self._raw_atomic_params[param_key][self._types_to_index[type_key]]
                    self.atomic_params[type_key] = these_atomic_params
            
            # Symmetrize the parameter dictionaries for convenience when making parameter arrays #
            for key in list(self.pair_params.keys()):
                self.pair_params[(key[1], key[0])] = self.pair_params[key]
            for key in list(self.angle_params.keys()):
                self.angle_params[(key[2], key[1], key[0])] = self.angle_params[key]

    def evaluate(self, cm: CoordinateManager, topology: Topology, params: Parameterizer, reset_grads: bool=False):
        # Get all intermolecular and intramolecular pairs, dists, and vectors inside long-range cutoff #
        pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs(topology, reset_grads=reset_grads)
        
        # Get pairs, dists, and vectors for exclusion list (needed to remove their contribution from long-range interactions) #
        pairs_excl = pairs[topology.all_intramolecular_pairs, :]
        pairs_excl_i_a = pairs_excl[:, 0]
        pairs_excl_j_a = pairs_excl[:, 1]
        dists_excl = dists[topology.all_intramolecular_pairs]
        dist_vecs_excl = dist_vecs[topology.all_intramolecular_pairs]

        # Get pairs, dists, and vectors for real-space potential #
        pairs_lr = pairs[topology.all_intermolecular_pairs, :]
        pairs_lr_i_a = pairs_lr[:, 0]
        pairs_lr_j_a = pairs_lr[:, 1]
        dists_lr = dists[topology.all_intermolecular_pairs]
        dist_vecs_lr = dist_vecs[topology.all_intermolecular_pairs]

        if topology.angle_pairs.numel() > 0:
            angles = computeAngleFromVecs(dist_vecs[topology.angle_pairs[0]], dist_vecs[topology.angle_pairs[1]])

        # All pairs forming an angle #
        pairs_angles_p = topology.angle_pairs.T.flatten()

        # Electric Multipoles #
        mono = params.get_atomic_parameters('mono')
        natoms = torch.tensor(mono.size(0), device=pairs.device)

        # Lennard-Jones parameters
        b_ij_disp_vdw_p = params.get_pair_parameters_with_optional_combination_rule(
            'b_disp', topology.all_intermolecular_pairs, pairs_lr
        )
        C6_ij_disp_vdw_p = params.get_pair_parameters_with_optional_combination_rule(
            'C6_disp', topology.all_intermolecular_pairs, pairs_lr
        )

        r_eq = params.get_pair_parameters('r_eq', topology.bonded_pairs)
        k_b_p = params.get_pair_parameters('k_b', topology.bonded_pairs)
        theta_eq = params.get_angle_parameters('theta_eq', topology.angle_atoms)
        k_theta = params.get_angle_parameters('k_theta', topology.angle_atoms)
        
        # Find appropriate ewald parameters. This should really be done by the CM.
        self.alpha_ewald = torch.sqrt(-torch.log10(2 * self.ewald_tolerance)) / self.cutoff_ewald
        self.k_max = 50
        for i in range(2, 50):
            error_estimate = (i * torch.sqrt(cm.box_lengths[0] * self.alpha_ewald) / 20.0) * torch.exp(-torch.pi * torch.pi * i * i / (cm.box_lengths[0] * self.alpha_ewald * cm.box_lengths[0] * self.alpha_ewald))
            if error_estimate < self.ewald_tolerance:
                self.k_max = i
                break
        erfc_damps = computeDampFactorsErfc(dists_lr, self.alpha_ewald) # direct space
        erf_damps = -computeDampFactorsErf(dists_excl, self.alpha_ewald)
        # ^^^ for removing excluded interactions that are implicitly included in long-range summation
        # The reciprocal space calculation uses an erf(alpha*r) damping so the above is -erf(alpha*r)

        mono_lr = mono
        # HERE: Implement the long_range_potential_rank_0 kernel and finish writing everything else as well.

        # Get reciprocal space and self contributions to field variables
        # and corresponding electrostatic interactions.
        ewald_potential, ewald_field, ewald_field_gradient = long_range_potential_rank_0(cm.coords, mono_lr, dipo_lr, quad_lr, cm.box, self.alpha_ewald, self.k_max)
        ene_ewald = 0.5 * (
            torch.einsum("n,n->", mono_lr, ewald_potential)
        )
        
        # Real Space Electrostatic Interactions #
        # TOOD: Just hard-code the implementation here. Need to get erfc damped interactions for the actual pairs
        # and erf-damped with the exluded list.
        elec_point_pairwise = torch.bmm(multipoles_real_j_p.unsqueeze(1), edata_point_pairwise).flatten()
        #elec_point_excl_pairwise = ...

        ene_perm_elec = 0.5 * (
            torch.sum(elec_point_pairwise)
        )
        ene_perm_elec = ene_perm_elec + 0.5 * torch.sum(elec_point_excl_pairwise)

        # NOTE(JOE): There is a problem with the gradients here when induced
        # fields are included. Basically, the partial derivatives of the induced
        # multipoles with respect to the cartesian coordinates are needed for the
        # FD morse derivatives. Unfortunately, if gradient tracking is on when the
        # polarization equations are solved, then things become very slow and
        # use a lot of memory (but the gradients are right!). If we have gradient
        # tracking off then everything is much more efficient but the FD morse
        # gradients are wrong. So, we need to compute the field gradients
        # due to the induced multipoles and properly incorporate them into the
        # pytorch computational graph. This is possible, but I am going to
        # figure that out once we are in a better position to actually run MD.
        ene_bonds = torch.zeros(1, dtype=dists.dtype, device=dists.device)
        if topology.bonded_pairs.numel() > 0:
            # TODO: Change to a harmonic bonding potential
            # morse-bond
            ene_bond_list = computeMorseBondPotential(dists[topology.bonded_pairs], re_fd_p, D_p, beta_fd_p)
            ene_bonds = torch.sum(ene_bond_list)

        ene_angles = torch.zeros(1, dtype=dists.dtype, device=dists.device)
        if topology.angle_atoms.numel() > 0:
            # TODO: Change to a harmonic angle potential
            # angles
            ene_angles_list = computeCosAnglePotential(
                angles, theta_eq, k_theta
            )
            ene_angles = torch.sum(ene_angles_list)

        # dispersion
        # TODO: Change to Leannrd-Jones potential
        lj_pairwise = computeDispersionFromPairs(
            dists_lr,
            C6_ij_disp_vdw_p, b_ij_disp_vdw_p,
            switch_lr
        )
        ene_lj = torch.sum(lj_pairwise) / 2

        # TODO: Change to long-range LJ correction (which might be identical actually)
        ene_lj_lr = torch.tensor(0.0)
        ene_lj_lr = compute_long_range_dispersion_correction(
            C6_ij_disp_vdw_p, self.cutoff_ewald,
            torch.tensor(cm.coords.size(0)), cm.box_volume
        )

        ene_tot = ene_perm_elec + ene_lj + ene_lj_lr + ene_bonds + ene_angles + ene_ewald
        energies = {
            "perm_elec": ene_perm_elec,
            "deformation": ene_bonds + ene_angles,
            "bond": ene_bonds,
            "angle": ene_angles,
            "ewald": ene_ewald,
            "lj": ene_lj,
            "lj_lr_correction": ene_lj_lr,
            "total": ene_tot
        }

        return energies