import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System

class ExchangePolarizationCMM(Term):
    def __init__(self):
        super().__init__()
    
    @property
    def param_data(self):
        return [('multipoles_xpol', ParameterType.Atomic, CutoffType.NB_Short)]
    
    @property
    def outputs(self):
        return ['V_xpol']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        pairs_short = system.storage.get('pairs_short')
        multipoles_xpol = system.storage.get('multipoles_xpol')
        xpol_interaction_tensor_short = system.storage.get('xpol_interaction_tensor_short')
        switch_short = system.storage.get('switch_short')

        multipoles_xpol_i_p = multipoles_xpol[pairs_short[:, 0]]
        multipoles_xpol_j_p = multipoles_xpol[pairs_short[:, 1]]
        xpol_pairwise = torch.bmm(multipoles_xpol_j_p.unsqueeze(1), torch.bmm(xpol_interaction_tensor_short, multipoles_xpol_i_p.unsqueeze(2))).flatten()

        return {'V_xpol': 0.5 * torch.sum(xpol_pairwise * switch_short)}