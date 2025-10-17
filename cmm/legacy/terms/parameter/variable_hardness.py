import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...bonded_parameter_functions import computeAngleFromVecs, computeHardnessChangeBond, computeHardnessChangeBondBond, computeHardnessChangeAngle

class VariableHardness(Term):
    def __init__(self):
        super().__init__()
    
    @property
    def param_data(self):
        return [
            ('r_eq', ParameterType.Pair, CutoffType.B_Bond),
            ('k_hardness_b', ParameterType.Pair, CutoffType.B_Bond),
            ('k_hardness_bb', ParameterType.PairPair, CutoffType.B_Angle),
            ('theta_eq', ParameterType.Angle, CutoffType.B_Angle),
            ('k_hardness_angle', ParameterType.Angle, CutoffType.B_Angle),
        ]
    
    @property
    def outputs(self):
        return []

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):

        eta = system.storage.get('eta')
        hardness_product = torch.ones_like(eta)
        if system.topology.angle_atoms.numel() > 0:
            bonded_pair_indices = system.storage.get('bonded_pair_indices')
            angle_pair_indices_ij = system.storage.get('angle_pair_indices_ij')
            angle_pair_indices_jk = system.storage.get('angle_pair_indices_jk')
            
            r_eq = system.parameterizer.get_pair_parameters('r_eq', bonded_pair_indices)
            k_hardness_b = system.parameterizer.get_pair_parameters('k_hardness_b', bonded_pair_indices)
            r_eq_bb_1 = system.parameterizer.get_pair_parameters('r_eq', angle_pair_indices_ij)
            r_eq_bb_2 = system.parameterizer.get_pair_parameters('r_eq', angle_pair_indices_jk)
            k_hardness_bb_1 = system.parameterizer.get_pair_pair_parameters('k_hardness_bb', angle_pair_indices_ij, angle_pair_indices_jk)
            k_hardness_bb_2 = system.parameterizer.get_pair_pair_parameters('k_hardness_bb', angle_pair_indices_jk, angle_pair_indices_ij)
            theta_eq = system.parameterizer.get_angle_parameters('theta_eq', system.topology.angle_atoms)
            k_hardness_angle = system.parameterizer.get_angle_parameters('k_hardness_angle', system.topology.angle_atoms)

            angles = computeAngleFromVecs(distance_vecs[angle_pair_indices_ij], distance_vecs[angle_pair_indices_jk])
            dists_bonded = dists[bonded_pair_indices]

            hardness_change_b = computeHardnessChangeBond(dists_bonded, r_eq, k_hardness_b)
            hardness_change_bb_1, hardness_change_bb_2 = computeHardnessChangeBondBond(
                dists[angle_pair_indices_ij], dists[angle_pair_indices_jk],
                r_eq_bb_1, r_eq_bb_2, k_hardness_bb_1, k_hardness_bb_2
            )
            hardness_change_angle = computeHardnessChangeAngle(angles, theta_eq, k_hardness_angle)

            # NOTE(JOE): We only accumulate some of the bond-bond terms since this basically assumes that all of
            # the hardness change is on the second atom of the bond. i.e. just the H atoms in water.
            # This is yet another reason not to like this piece of the model. There should be a better way.

            # Cannot do in-place operations or else computational graphs breaks. Sad.
            hardness_product = hardness_product.scatter_reduce(0, system.topology.bonded_atoms[1], hardness_change_b, reduce="prod")
            hardness_product = hardness_product.scatter_reduce(0, system.topology.angle_atoms.t()[0], hardness_change_bb_1, reduce="prod")
            hardness_product = hardness_product.scatter_reduce(0, system.topology.angle_atoms.t()[2], hardness_change_bb_2, reduce="prod")

            eta *= hardness_product
            eta.scatter_add_(0, system.topology.angle_atoms.t()[0], hardness_change_angle)
            eta.scatter_add_(0, system.topology.angle_atoms.t()[2], hardness_change_angle)
        
        system.storage.add('eta', 2 * eta)
        # ^^^ Polarization works with the factor of 2 in there. This is basically just a historical holdover
        # due to an inconsistency in how the parameters were originally fit and later re-implemented.
        # Whenever the hardness model gets updated we should clean this up.

        return {}