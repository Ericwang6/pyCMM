import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...bonded import computeFieldDependentMorseParams

class FieldDependentMorseParams(Term):
    def __init__(self):
        super().__init__()
    
    @property
    def param_data(self):
        return [
            ('r_eq', ParameterType.Pair, CutoffType.B_Bond),
            ('k_b', ParameterType.Pair, CutoffType.B_Bond),
            ('D', ParameterType.Pair, CutoffType.B_Bond),
            ('dip_deriv_1', ParameterType.Pair, CutoffType.B_Bond),
            ('dip_deriv_2', ParameterType.Pair, CutoffType.B_Bond),
            ('ct_slope_1', ParameterType.Pair, CutoffType.B_Bond),
            ('ct_slope_2', ParameterType.Pair, CutoffType.B_Bond),
        ]
    
    @property
    def outputs(self):
        return []

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        beta_morse = torch.zeros(int(system.topology.bonded_atoms.numel() / 2), device=system.device, dtype=torch.get_default_dtype())
        r_eq_morse = torch.zeros(int(system.topology.bonded_atoms.numel() / 2), device=system.device, dtype=torch.get_default_dtype())
        if system.topology.bonded_atoms.numel() > 0:
            bonded_pair_indices = system.neighbor_list.get_pair_indices(system.topology.bonded_atoms.T)
            dists_bonded = dists[bonded_pair_indices]
            distance_vecs_bonded = distance_vecs[bonded_pair_indices]
            r_eq = system.parameterizer.get_pair_parameters('r_eq', bonded_pair_indices)
            k_b_p = system.parameterizer.get_pair_parameters('k_b', bonded_pair_indices)
            D_p = system.parameterizer.get_pair_parameters('D', bonded_pair_indices)
            dip_deriv_1_p = system.parameterizer.get_pair_parameters('dip_deriv_1', bonded_pair_indices)
            dip_deriv_2_p = system.parameterizer.get_pair_parameters('dip_deriv_2', bonded_pair_indices)
            ct_slope_1_p = system.parameterizer.get_pair_parameters('ct_slope_1', bonded_pair_indices)
            ct_slope_2_p = system.parameterizer.get_pair_parameters('ct_slope_2', bonded_pair_indices)

            electric_field_data = system.storage.get('electric_field_data')
            elec_field = electric_field_data.mul(torch.tensor([1, -1, -1, -1, -1, -1, -1, -1, -1, -1], device=pairs.device).reshape(1, -1))[:, 1:4]
            dq_a = system.storage.get('dq_a')
            print(elec_field)
            r_eq_morse, beta_morse = computeFieldDependentMorseParams(
                dists_bonded, distance_vecs_bonded,
                k_b_p, D_p, r_eq, dip_deriv_1_p, dip_deriv_2_p,
                ct_slope_1_p, ct_slope_2_p,
                (elec_field)[system.topology.bonded_atoms[1]],
                dq_a[system.topology.bonded_atoms[1]]
            )
            
        system.storage.add('beta_morse', beta_morse)
        system.storage.add('r_eq_morse', r_eq_morse)

        return {}