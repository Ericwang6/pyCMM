import torch
from ..term import Term, InteractionType
from ...system import System
from ...dispersion import computeLennardJonesFromPairs

class LennardJones(Term):
    @property
    def interaction_type(self):
        return InteractionType.Dispersion
    
    @property
    def params(self):
        return ['sigma_lj', 'eps_lj']
    
    @property
    def outputs(self):
        return ['V_lj']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        included_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.included_pairs)
        pairs_lr = pairs[included_pair_indices, :]
        dists_lr = dists[included_pair_indices]

        eps_ij_vdw_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'eps_lj', included_pair_indices, pairs_lr
        )
        sigma_ij_vdw_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'sigma_lj', included_pair_indices, pairs_lr
        )

        lj_pairwise = computeLennardJonesFromPairs(
            dists_lr, sigma_ij_vdw_p, eps_ij_vdw_p
        )
        V_lj = torch.sum(lj_pairwise) / 2
        return {'V_lj': V_lj}