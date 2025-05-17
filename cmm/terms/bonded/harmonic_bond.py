import torch
from ..term import Term, InteractionType
from ...system import System
from ...bonded import computeHarmonicBondPotential

class HarmonicBond(Term):
    @property
    def interaction_type(self):
        return InteractionType.Bonded
    
    @property
    def params(self):
        return ['k_bond', 'r_eq']
    
    @property
    def outputs(self):
        return ['V_bond']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        bonded_pair_indices = system.neighbor_list.get_pair_indices(system.topology.bonded_atoms.T)
        k_bond = system.parameterizer.get_pair_parameters('k_bond', bonded_pair_indices)
        r_eq = system.parameterizer.get_pair_parameters('r_eq', bonded_pair_indices)
        
        V_bond_pairs = computeHarmonicBondPotential(dists[bonded_pair_indices], r_eq, k_bond)
        return {'V_bond': torch.sum(V_bond_pairs)}
    

