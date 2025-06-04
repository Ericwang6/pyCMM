import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System

class MultipolarElectrostatics2(Term):
    def __init__(self, get_fields: bool=True):
        super().__init__()
        self.get_fields = get_fields
    
    @property
    def param_data(self):
        return [('multipoles_real', ParameterType.Atomic, CutoffType.NB_Medium)]
    
    @property
    def outputs(self):
        return ['V_elec_direct']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        pairs_medium = system.storage.get('pairs_medium')
        multipoles_real = system.storage.get('multipoles_real')
        direct_field_tensor_medium = system.storage.get('direct_field_tensor_medium')

        multipoles_real_i_p = multipoles_real[pairs_medium[:, 0]]
        multipoles_real_j_p = multipoles_real[pairs_medium[:, 1]]
        
        edata_point_pairwise = torch.bmm(direct_field_tensor_medium, multipoles_real_i_p.unsqueeze(2))
        elec_point_pairwise = torch.bmm(multipoles_real_j_p.unsqueeze(1), edata_point_pairwise).flatten()

        if self.get_fields:
            electric_field_data = system.storage.get('electric_field_data')
            electric_field_data = electric_field_data.scatter_add(0, pairs_medium[:, 1].unsqueeze(1).expand(-1, 10), edata_point_pairwise.squeeze(2))
            system.storage.add('electric_field_data', electric_field_data)
        return {'V_elec_direct': 0.5 * torch.sum(elec_point_pairwise)}

class ExcludedMultipolarElectrostatics2(Term):
    def __init__(self, get_fields: bool=True):
        super().__init__()
        self.get_fields = get_fields
    
    @property
    def param_data(self):
        return [('multipoles_real', ParameterType.Atomic, CutoffType.NB_Ewald)]
    
    @property
    def outputs(self):
        return ['V_elec_excl']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        pairs_excl = system.storage.get('pairs_excl')
        multipoles_real = system.storage.get('multipoles_real')
        direct_field_tensor_excl = system.storage.get('direct_field_tensor_excl')

        multipoles_real_i_p = multipoles_real[pairs_excl[:, 0]]
        multipoles_real_j_p = multipoles_real[pairs_excl[:, 1]]
        
        edata_point_pairwise = torch.bmm(direct_field_tensor_excl, multipoles_real_i_p.unsqueeze(2))
        elec_point_pairwise = torch.bmm(multipoles_real_j_p.unsqueeze(1), edata_point_pairwise).flatten()

        if self.get_fields:
            electric_field_data = system.storage.get('electric_field_data')
            electric_field_data = electric_field_data.scatter_add(0, pairs_excl[:, 1].unsqueeze(1).expand(-1, 10), edata_point_pairwise.squeeze(2))
            system.storage.add('electric_field_data', electric_field_data)

        return {'V_elec_excl': 0.5 * torch.sum(elec_point_pairwise)}