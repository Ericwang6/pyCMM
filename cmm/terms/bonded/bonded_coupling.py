import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...bonded import computeBondAngleCoupling, computeBondBondCoupling, computeAngleFromVecs

class BondedCouplingCMM(Term):
    
    @property
    def param_data(self):
        return [('k_ba', ParameterType.PairAngle, CutoffType.B_Angle), ('k_bb', ParameterType.PairPair, CutoffType.B_Angle), ('theta_eq', ParameterType.Angle, CutoffType.B_Angle)]
    
    @property
    def outputs(self):
        return ['V_bond_angle', 'V_bond_bond']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        if system.topology.angle_atoms.numel() > 0:
            angle_pair_indices_ij = system.storage.get('angle_pair_indices_ij')
            angle_pair_indices_jk = system.storage.get('angle_pair_indices_jk')
            angle_pairs_flat_p = torch.stack((angle_pair_indices_ij, angle_pair_indices_jk), dim=1).flatten()
            r_eq_bb_1 = system.parameterizer.get_pair_parameters('r_eq', angle_pair_indices_ij)
            r_eq_bb_2 = system.parameterizer.get_pair_parameters('r_eq', angle_pair_indices_jk)
            r_eq_ba = torch.stack((r_eq_bb_1, r_eq_bb_2), dim=1).flatten()
            k_ba = system.parameterizer.get_pair_angle_parameters('k_ba', angle_pairs_flat_p, system.topology.angle_atoms)
            k_bb = system.parameterizer.get_pair_pair_parameters('k_bb',  angle_pair_indices_ij, angle_pair_indices_jk)
            
            theta_eq = system.parameterizer.get_angle_parameters('theta_eq', system.topology.angle_atoms)
            angles = computeAngleFromVecs(distance_vecs[angle_pair_indices_ij], distance_vecs[angle_pair_indices_jk])

            # bond-angle couplings #
            ene_bas_list = computeBondAngleCoupling(
                dists[angle_pairs_flat_p], r_eq_ba,
                angles.repeat_interleave(2), theta_eq.repeat_interleave(2),
                k_ba
            )
            # bond-bond couplings #
            ene_bbs_list = computeBondBondCoupling(
                dists[angle_pair_indices_ij], dists[angle_pair_indices_jk],
                r_eq_bb_1, r_eq_bb_2, k_bb
            )
            return {
                'V_bond_angle': torch.sum(ene_bas_list),
                'V_bond_bond': torch.sum(ene_bbs_list),
            }