import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System

class MultipolarChargePenetration(Term):
    def __init__(self, get_fields: bool=True):
        super().__init__()
        self.get_fields = get_fields
    
    @property
    def param_data(self):
        return [('multipoles_cp', ParameterType.Atomic, CutoffType.NB_Short)]
    
    @property
    def outputs(self):
        return ['V_elec_cp']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        pairs_short = system.storage.get('pairs_short')
        multipoles_cp = system.storage.get('multipoles_cp')
        multipoles_Z = system.storage.get('multipoles_Z')
        cp_interaction_tensor_short = system.storage.get('cp_interaction_tensor_short')
        cp_field_tensor_short_i = system.storage.get('cp_field_tensor_short_i')
        cp_field_tensor_short_j = system.storage.get('cp_field_tensor_short_j')
        switch_short = system.storage.get('switch_short')

        multipoles_cp_i_p = multipoles_cp[pairs_short[:, 0]]
        multipoles_cp_j_p = multipoles_cp[pairs_short[:, 1]]
        multipoles_Z_i_p = multipoles_Z[pairs_short[:, 0]]
        multipoles_Z_j_p = multipoles_Z[pairs_short[:, 1]]
        cp_pairwise_ss = torch.bmm(
            multipoles_cp_j_p.unsqueeze(1),
            torch.bmm(cp_interaction_tensor_short, multipoles_cp_i_p.unsqueeze(2))
        ).flatten()
        edata_cs_pairwise_ij = torch.bmm(cp_field_tensor_short_i, multipoles_cp_i_p.unsqueeze(2))
        elec_cs_pairwise_ji = torch.bmm(multipoles_cp_j_p.unsqueeze(1), torch.bmm(cp_field_tensor_short_j, multipoles_Z_i_p.unsqueeze(2))).flatten()
        elec_cs_pairwise_ij = torch.bmm(multipoles_Z_j_p.unsqueeze(1), edata_cs_pairwise_ij).flatten()

        if self.get_fields:
            electric_field_data = system.storage.get('electric_field_data')
            electric_field_data = electric_field_data.scatter_add(0, pairs_short[:, 1].unsqueeze(1).expand(-1, 10), edata_cs_pairwise_ij.squeeze(2))
            system.storage.add('electric_field_data', electric_field_data)

        return {'V_elec_cp': 0.5 * torch.sum((cp_pairwise_ss + elec_cs_pairwise_ji + elec_cs_pairwise_ij) * switch_short)}