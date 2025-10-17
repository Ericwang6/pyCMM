import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System

class MultipolarChargeTransfer(Term):
    def __init__(self):
        super().__init__()
    
    @property
    def param_data(self):
        return [
            ('multipoles_ct_acc', ParameterType.Atomic, CutoffType.NB_Short),
            ('multipoles_ct_don', ParameterType.Atomic, CutoffType.NB_Short)
        ]
    
    @property
    def outputs(self):
        return ['V_ct_direct']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        pairs_short = system.storage.get('pairs_short')
        multipoles_ct_acc = system.storage.get('multipoles_ct_acc')
        multipoles_ct_don = system.storage.get('multipoles_ct_don')
        ct_interaction_tensor_short = system.storage.get('ct_interaction_tensor_short')
        switch_short = system.storage.get('switch_short')

        multipoles_ct_acc_i_p = multipoles_ct_acc[pairs_short[:, 0]]
        multipoles_ct_acc_j_p = multipoles_ct_acc[pairs_short[:, 1]]
        multipoles_ct_don_i_p = multipoles_ct_don[pairs_short[:, 0]]
        multipoles_ct_don_j_p = multipoles_ct_don[pairs_short[:, 1]]
        ct_pairwise_ij = torch.bmm(multipoles_ct_don_j_p.unsqueeze(1), torch.bmm(ct_interaction_tensor_short, multipoles_ct_acc_i_p.unsqueeze(2))).flatten()
        ct_pairwise_ji = torch.bmm(multipoles_ct_acc_j_p.unsqueeze(1), torch.bmm(ct_interaction_tensor_short, multipoles_ct_don_i_p.unsqueeze(2))).flatten()

        return {'V_ct_direct': 0.5 * torch.sum((ct_pairwise_ij + ct_pairwise_ji) * switch_short)}