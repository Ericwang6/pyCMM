import torch, math
from .electrostatics import computeDampFactorsErfc, computeDampFactorsErf
from .ewald import long_range_potential_rank_0
from .dispersion import computeLennardJonesFromPairs, compute_long_range_lennard_jones_correction
from .coordinate_manager import CoordinateManager
from .parameters import Parameterizer
from .topology import Topology
from .terms import *
from .units import *
from .switching_functions import switch_543
from .force_field import ForceField
from .timing_context import TimingContext

class SPCfw(ForceField):
    def __init__(self, ewald_tolerance: torch.Tensor=torch.tensor(1e-6, dtype=torch.float64)) -> None:
        super().__init__()
        # These are local indices for the atom types, not the actual
        # atom type indices which are decided by the Parameterizer.
        self._types_to_index = {
            "O_water": 0, "H_water": 1
        }
        self.ewald_tolerance = ewald_tolerance if torch.is_tensor(ewald_tolerance) else torch.tensor(ewald_tolerance, dtype=torch.float64)
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
                "k_theta": torch.tensor([75.90 / HARTREE2KCAL]),
                "theta_eq": torch.tensor([113.24 * math.pi / 180.0]),
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
        with TimingContext("ff/cm"):
            pairs, dists, dist_vecs = cm.get_distances_vectors_and_pairs(topology, reset_grads=reset_grads)
            cutoff_ewald = cm.cutoff
        
        with TimingContext("ff/find_ewald_k_max"):
            if self.alpha_ewald is None:
                # Find appropriate ewald parameters. This should really be done by the CM.
                self.alpha_ewald = torch.sqrt(-torch.log10(2 * self.ewald_tolerance)) / cutoff_ewald
                self.k_max = 50
                for i in range(2, 50):
                    error_estimate = (i * torch.sqrt(cm.box_lengths[0] * self.alpha_ewald) / 20.0) * torch.exp(-torch.pi * torch.pi * i * i / (cm.box_lengths[0] * self.alpha_ewald * cm.box_lengths[0] * self.alpha_ewald))
                    if error_estimate < self.ewald_tolerance:
                        self.k_max = i
                        break
        with TimingContext("ff/build_params"):
            params.rebuild(pairs, self.atomic_params, self.pair_params, {}, {}, self.angle_params)

        with TimingContext("ff/get_subpairs"):
            # Get pairs, dists, and vectors for exclusion list (needed to remove their contribution from long-range interactions) #
            excluded_pair_indices = cm.neighbor_list.get_pair_indices(cm.neighbor_list.excluded_pairs)
            included_pair_indices = cm.neighbor_list.get_pair_indices(cm.neighbor_list.included_pairs)
            pairs_excl = pairs[excluded_pair_indices, :]
            pairs_excl_i_a = pairs_excl[:, 0]
            pairs_excl_j_a = pairs_excl[:, 1]
            dists_excl = dists[excluded_pair_indices]

            # Get pairs, dists, and vectors for real-space potential #
            pairs_lr = pairs[included_pair_indices, :]
            pairs_lr_i_a = pairs_lr[:, 0]
            pairs_lr_j_a = pairs_lr[:, 1]
            dists_lr = dists[included_pair_indices]

        with TimingContext("ff/get_parameters"):
            # Electric Multipoles #
            mono = params.get_atomic_parameters('mono')
            natoms = torch.tensor(mono.size(0), device=pairs.device)

            with TimingContext("ff/get_parameters/LJ_params"):
                # Lennard-Jones parameters
                eps_ij_vdw_p = params.get_pair_parameters_with_optional_combination_rule(
                    'eps_lj', included_pair_indices, pairs_lr
                )
                sigma_ij_vdw_p = params.get_pair_parameters_with_optional_combination_rule(
                    'sigma_lj', included_pair_indices, pairs_lr
                )

            with TimingContext("ff/get_parameters/bonded_params"):
                bonded_pair_indices = cm.neighbor_list.get_pair_indices(topology.bonded_atoms.T)
                angle_pair_indices_ij = cm.neighbor_list.get_pair_indices(topology.angle_atoms[:, 0:2])
                angle_pair_indices_jk = cm.neighbor_list.get_pair_indices(topology.angle_atoms[:, 1:].flip(1))
                r_eq = params.get_pair_parameters('r_eq', bonded_pair_indices)
                k_b_p = params.get_pair_parameters('k_b', bonded_pair_indices)
                theta_eq = params.get_angle_parameters('theta_eq', topology.angle_atoms)
                k_theta = params.get_angle_parameters('k_theta', topology.angle_atoms)
        
        with TimingContext("ff/elec"):
            erfc_damps = computeDampFactorsErfc(dists_lr, self.alpha_ewald)
            erf_damps = -computeDampFactorsErf(dists_excl, self.alpha_ewald)

            with TimingContext("ff/elec/ewald"):
                # Get reciprocal space and self contributions to field variables
                # and corresponding electrostatic interactions.
                mono_lr = mono
                ewald_potential = long_range_potential_rank_0(cm.coords, mono_lr, cm.box, self.alpha_ewald, self.k_max)
                elec_point_excl_pairwise = erf_damps[0, :] * mono[pairs_excl_i_a] * mono[pairs_excl_j_a] / dists_excl
                ene_ewald = 0.5 * (
                    torch.einsum("n,n->", mono_lr, ewald_potential) +
                    torch.sum(elec_point_excl_pairwise)
                )
            with TimingContext("ff/elec/real"):
                # Real Space Electrostatic Interactions #
                elec_point_pairwise = erfc_damps[0, :] * mono_lr[pairs_lr_i_a] * mono_lr[pairs_lr_j_a] / dists_lr
                ene_perm_elec = 0.5 * (
                    torch.sum(elec_point_pairwise)
                )
        with TimingContext("ff/bond"):
            ene_bond_list = computeHarmonicBondPotential(dists[bonded_pair_indices], r_eq, k_b_p)
            ene_bonds = torch.sum(ene_bond_list)
        with TimingContext("ff/angles"):
            angles = computeAngleFromVecs(dist_vecs[angle_pair_indices_ij], dist_vecs[angle_pair_indices_jk])
            ene_angles_list = computeHarmonicAnglePotential(
                angles, theta_eq, k_theta
            )
            ene_angles = torch.sum(ene_angles_list)

        with TimingContext("ff/LJ"):
            # dispersion
            lj_pairwise = computeLennardJonesFromPairs(
                dists_lr, sigma_ij_vdw_p, eps_ij_vdw_p
            )
            ene_lj = torch.sum(lj_pairwise) / 2

            # NOTE(JOE): Technically what we are doing is slightly different than
            # the derived LRC formula since we do not include all pairs,
            # we compute the average LJ parameters respecting exclusions.
            # Should really re-derive the formula for the case of exclusions.
            ene_lj_lr = compute_long_range_lennard_jones_correction(
                sigma_ij_vdw_p, eps_ij_vdw_p, cutoff_ewald,
                natoms, cm.box_volume
            )
            ene_lj = ene_lj + ene_lj_lr

        ene_tot = ene_perm_elec + ene_lj + ene_bonds + ene_angles + ene_ewald
        energies = {
            "perm_elec": ene_perm_elec,
            "total_elec": ene_perm_elec + ene_ewald,
            "deformation": ene_bonds + ene_angles,
            "bond": ene_bonds,
            "angle": ene_angles,
            "ewald": ene_ewald,
            "lj": ene_lj,
            "lj_lr_correction": ene_lj_lr,
            "total": ene_tot
        }

        return energies