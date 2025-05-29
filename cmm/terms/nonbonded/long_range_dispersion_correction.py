import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...dispersion import compute_long_range_lennard_jones_correction

class LongRangeLennardJonesCorrection(Term):
    
    @property
    def param_data(self):
        return [('sigma_lj', ParameterType.Pair, CutoffType.NB_Long), ('eps_lj', ParameterType.Pair, CutoffType.NB_Long)]
    
    @property
    def outputs(self):
        return ['V_lj_lr']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        included_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.included_pairs)
        pairs_lr = pairs[included_pair_indices, :]

        eps_ij_vdw_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'eps_lj', included_pair_indices, pairs_lr
        )
        sigma_ij_vdw_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'sigma_lj', included_pair_indices, pairs_lr
        )
        print(sigma_ij_vdw_p)
        V_lj_lr = compute_long_range_lennard_jones_correction(
            sigma_ij_vdw_p, eps_ij_vdw_p, system.neighbor_list.cutoff,
            system.neighbor_list.natoms, system.box_volume
        )
        return {'V_lj_lr': V_lj_lr}