import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...dispersion import computeDispersionFromPairs

class TTDispersionC6(Term):
    def __init__(self, use_switching: bool, switching_start: float):
        self.use_switching = use_switching
        self.switching_start = switching_start
    
    @property
    def param_data(self):
        return [('C6_disp', ParameterType.Pair, CutoffType.NB_Long), ('b_disp', ParameterType.Pair, CutoffType.NB_Long)]
    
    @property
    def outputs(self):
        return ['V_disp']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        included_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.included_pairs)
        pairs_lr = pairs[included_pair_indices, :]
        dists_lr = dists[included_pair_indices]

        b_ij_disp_vdw_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'b_disp', included_pair_indices, pairs_lr
        )
        C6_ij_disp_vdw_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'C6_disp', included_pair_indices, pairs_lr
        )

        disp_pairwise = computeDispersionFromPairs(
            dists_lr,
            C6_ij_disp_vdw_p, b_ij_disp_vdw_p
        )
        V_disp = torch.sum(disp_pairwise) / 2
        return {'V_disp': V_disp}