import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from torch_scatter import segment_csr

class ManyBodyChargeTransfer(Term):
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
        return []

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        included_pair_indices_short = system.storage.get('included_pair_indices_short')
        pairs_short = system.storage.get('pairs_short')
        multipoles_ct_acc = system.storage.get('multipoles_ct_acc')
        multipoles_ct_don = system.storage.get('multipoles_ct_don')
        ct_interaction_tensor_short = system.storage.get('ct_interaction_tensor_short')
        switch_short = system.storage.get('switch_short')
        eps = system.parameterizer.get_pair_parameters_with_optional_combination_rule('eps', included_pair_indices_short, pairs_short)

        multipoles_ct_acc_i_p = multipoles_ct_acc[pairs_short[:, 0]]
        multipoles_ct_acc_j_p = multipoles_ct_acc[pairs_short[:, 1]]
        multipoles_ct_don_i_p = multipoles_ct_don[pairs_short[:, 0]]
        multipoles_ct_don_j_p = multipoles_ct_don[pairs_short[:, 1]]
        drInvDamp_ct = ct_interaction_tensor_short[:, 0, 0].flatten()
        dq_forward = multipoles_ct_don_i_p[:, 0] * multipoles_ct_acc_j_p[:, 0] * drInvDamp_ct * eps
        dq_backward = multipoles_ct_acc_i_p[:, 0] * multipoles_ct_don_j_p[:, 0] * drInvDamp_ct * eps
        dq_pairwise = (dq_forward - dq_backward) * switch_short
        
        dq_a = torch.zeros(system.neighbor_list.natoms, device=pairs.device, requires_grad=True)
        dq_groups = torch.zeros(system.topology.n_pol_groups, device=pairs.device, requires_grad=True)
        dq_a = dq_a.scatter_add(0, pairs_short[:, 1], dq_pairwise)
        dq_groups = segment_csr(dq_a[system.topology.pol_group_indices_a], system.topology.pol_group_segment_indices, reduce='sum')

        system.storage.add('dq_a', dq_a)
        system.storage.add('dq_groups', dq_groups)

        return {}