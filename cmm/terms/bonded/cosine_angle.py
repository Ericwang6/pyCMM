import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...bonded import computeCosAnglePotential, computeAngleFromVecs

class CosineAngle(Term):
    
    @property
    def param_data(self):
        return [('k_angle', ParameterType.Angle, CutoffType.B_Angle), ('theta_eq', ParameterType.Angle, CutoffType.B_Angle)]
    
    @property
    def outputs(self):
        return ['V_angle']

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        if system.topology.angle_atoms.numel() > 0:
            angle_pair_indices_ij = system.storage.get('angle_pair_indices_ij')
            angle_pair_indices_jk = system.storage.get('angle_pair_indices_jk')
            theta_eq = system.parameterizer.get_angle_parameters('theta_eq', system.topology.angle_atoms)
            k_theta = system.parameterizer.get_angle_parameters('k_theta', system.topology.angle_atoms)
            angles = computeAngleFromVecs(distance_vecs[angle_pair_indices_ij], distance_vecs[angle_pair_indices_jk])

            V_angles = computeCosAnglePotential(
                angles, theta_eq, k_theta
            )
            return {'V_angle': torch.sum(V_angles)}
        return {'V_angle': torch.tensor(0.0, device=system.device)}