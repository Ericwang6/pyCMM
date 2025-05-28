import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...electrostatics import computeDampFactorsErf
from ...ewald import long_range_potential_rank_0, long_range_potential

class EwaldEnergy0(Term):
    def __init__(self, alpha: float, k_max: int):
        self.alpha = alpha
        self.k_max = k_max
        super().__init__()
    
    @property
    def param_data(self):
        return [('q', ParameterType.Atomic, CutoffType.NB_Ewald)]
    
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

class EwaldEnergy2(Term):
    def __init__(self, alpha: float, k_max: int):
        self.alpha = alpha
        self.k_max = k_max
        super().__init__()
    
    @property
    def param_data(self):
        return [
            ('q', ParameterType.Atomic, CutoffType.NB_Ewald),
            ('dipo', ParameterType.Atomic, CutoffType.NB_Ewald),
            ('quad', ParameterType.Atomic, CutoffType.NB_Ewald)
        ]
    
    @property
    def outputs(self):
        return ['V_elec_lr']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        q = system.storage.get('q')
        dipo = system.storage.get('dipo')
        quad = system.storage.get('quad')
        ewald_potential, ewald_field, ewald_field_gradient = long_range_potential(system.coords, q, dipo, quad, system.box, self.alpha, self.k_max)
        V_elec_lr = 0.5 * (
            torch.einsum("n,n->", q, ewald_potential) -
            torch.einsum("ni,ni->", dipo, ewald_field) -
            torch.einsum("nij,nij->", quad, ewald_field_gradient) / 3
        )

        system.storage.add('ewald_potential', ewald_potential)
        system.storage.add('ewald_field', ewald_field)
        system.storage.add('ewald_field_gradient', ewald_field_gradient)

        return {'V_elec_lr': V_elec_lr}