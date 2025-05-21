import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...bonded import computeMorseBondPotential

class MorseBond(Term):
    
    @property
    def param_data(self):
        return [('k_bond', ParameterType.Pair, CutoffType.B_Bond), ('r_eq', ParameterType.Pair, CutoffType.B_Bond)]
    
    @property
    def outputs(self):
        return ['V_bond']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        bonded_pair_indices = system.neighbor_list.get_pair_indices(system.topology.bonded_atoms.T)
        k_bond = system.parameterizer.get_pair_parameters('k_bond', bonded_pair_indices)
        r_eq = system.parameterizer.get_pair_parameters('r_eq', bonded_pair_indices)
        
        V_bond_pairs = computeMorseBondPotential(dists[bonded_pair_indices], r_eq, k_bond)
        return {'V_bond': torch.sum(V_bond_pairs)}