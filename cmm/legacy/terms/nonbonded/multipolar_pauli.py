import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System

class MultipolarPauli(Term):
    def __init__(self):
        super().__init__()
    
    @property
    def param_data(self):
        return [('multipoles_pauli', ParameterType.Atomic, CutoffType.NB_Short)]
    
    @property
    def outputs(self):
        return ['V_pauli']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        pairs_short = system.storage.get('pairs_short')
        multipoles_pauli = system.storage.get('multipoles_pauli')
        cp_interaction_tensor_short = system.storage.get('pauli_interaction_tensor_short')
        switch_short = system.storage.get('switch_short')

        multipoles_pauli_i_p = multipoles_pauli[pairs_short[:, 0]]
        multipoles_pauli_j_p = multipoles_pauli[pairs_short[:, 1]]
        pauli_pairwise = torch.bmm(
            multipoles_pauli_j_p.unsqueeze(1),
            torch.bmm(cp_interaction_tensor_short, multipoles_pauli_i_p.unsqueeze(2))
        ).flatten()

        return {'V_pauli': 0.5 * torch.sum(pauli_pairwise * switch_short)}