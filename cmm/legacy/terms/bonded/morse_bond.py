import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...bonded import computeMorseBondPotential

class MorseBond(Term):
    
    @property
    def param_data(self):
        return [
        ('D', ParameterType.Pair, CutoffType.B_Bond),
        ('beta_morse', ParameterType.Pair, CutoffType.B_Bond),
        ('r_eq_morse', ParameterType.Pair, CutoffType.B_Bond)
    ]
    
    @property
    def outputs(self):
        return ['V_bond']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        bonded_pair_indices = system.neighbor_list.get_pair_indices(system.topology.bonded_atoms.T)
        D_morse = system.parameterizer.get_pair_parameters('D', bonded_pair_indices)
        beta_morse = system.storage.get('beta_morse')
        r_eq_morse = system.storage.get('r_eq_morse')
        
        V_bond_pairs = computeMorseBondPotential(dists[bonded_pair_indices], r_eq_morse, D_morse, beta_morse)
        return {'V_bond': torch.sum(V_bond_pairs)}