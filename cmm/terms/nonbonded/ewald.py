import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...electrostatics import computeDampFactorsErf
from ...ewald import long_range_potential_rank_0

class EwaldEnergy0(Term):
    def __init__(self, alpha: float, k_max: int):
        self.alpha = alpha
        self.k_max = k_max
        super().__init__()
    
    @property
    def cutoff_type(self):
        return CutoffType.NB_Ewald
    
    @property
    def params(self):
        return [('q', ParameterType.Atomic)]
    
    @property
    def outputs(self):
        return ['V_elec_lr']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        excluded_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.excluded_pairs)
        pairs_excl_i_a = pairs[excluded_pair_indices, 0]
        pairs_excl_j_a = pairs[excluded_pair_indices, 1]
        dists_excl = dists[excluded_pair_indices]
        
        q = system.parameterizer.get_atomic_parameters('q')
        erf_damps = -computeDampFactorsErf(dists_excl, self.alpha)

        ewald_potential = long_range_potential_rank_0(system.coords, q, system.box, self.alpha, self.k_max)
        elec_point_excl_pairwise = erf_damps[0, :] * q[pairs_excl_i_a] * q[pairs_excl_j_a] / dists_excl
        V_elec_lr = 0.5 * (
            torch.einsum("n,n->", q, ewald_potential) +
            torch.sum(elec_point_excl_pairwise)
        )

        return {'V_elec_lr': V_elec_lr}