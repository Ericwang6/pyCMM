import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...dispersion import compute_long_range_lennard_jones_correction, compute_long_range_dispersion_correction

class LongRangeLennardJonesCorrection(Term):
    @property
    def param_data(self):
        return [('sigma_lj', ParameterType.Pair, CutoffType.NB_Long), ('eps_lj', ParameterType.Pair, CutoffType.NB_Long)]
    
    @property
    def outputs(self):
        return ['V_lj_lr']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        included_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.included_pairs)
        pairs_long = system.storage.get('pairs_long')

        eps_ij_vdw_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'eps_lj', included_pair_indices, pairs_long
        )
        sigma_ij_vdw_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'sigma_lj', included_pair_indices, pairs_long
        )

        V_lj_lr = compute_long_range_lennard_jones_correction(
            sigma_ij_vdw_p, eps_ij_vdw_p, system.neighbor_list.cutoff,
            system.neighbor_list.natoms, system.box_volume
        )
        return {'V_lj_lr': V_lj_lr}

class LongRangeC6DispersionCorrection(Term):
    @property
    def param_data(self):
        return [('C6_disp', ParameterType.Pair, CutoffType.NB_Long)]
    
    @property
    def outputs(self):
        return ['V_disp_lr']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        included_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.included_pairs)
        pairs_long = system.storage.get('pairs_long')

        C6_ij_disp_vdw_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'C6_disp', included_pair_indices, pairs_long
        )

        V_disp_lr = ene_disp_lr = compute_long_range_dispersion_correction(
                C6_ij_disp_vdw_p, system.neighbor_list.cutoff,
                system.neighbor_list.natoms, system.box_volume
            )
        return {'V_disp_lr': V_disp_lr}