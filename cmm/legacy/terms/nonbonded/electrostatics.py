import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...electrostatics import computeDampFactorsErfc

class ElectrostaticEnergy0(Term):
    def __init__(self, alpha: float):
        self.alpha = alpha
        super().__init__()
    
    @property
    def param_data(self):
        return [('q', ParameterType.Atomic, CutoffType.NB_Medium)]
    
    @property
    def outputs(self):
        return ['V_elec_direct']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        included_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.included_pairs)
        pairs_lr_i_a = pairs[included_pair_indices, 0]
        pairs_lr_j_a = pairs[included_pair_indices, 1]
        dists_lr = dists[included_pair_indices]
        
        q = system.parameterizer.get_atomic_parameters('q')
        erfc_damps = computeDampFactorsErfc(dists_lr, self.alpha)

        elec_point_pairwise = erfc_damps[0, :] * q[pairs_lr_i_a] * q[pairs_lr_j_a] / dists_lr
        V_elec_direct = 0.5 * (
            torch.sum(elec_point_pairwise)
        )

        return {'V_elec_direct': V_elec_direct}
