import torch
from ..term import Term, CutoffType, ParameterType
from ...system import System
from ...bonded import computeAngleFromVecs, computeChargeFluxBond, computeChargeFluxBondBond, computeChargeFluxAngle

class StoreChargeFluxCMM(Term):
    def __init__(self):
        super().__init__()
    
    @property
    def param_data(self):
        return [
            ('j_cf', ParameterType.Pair, CutoffType.B_Bond), ('r_eq', ParameterType.Pair, CutoffType.B_Bond),
            ('j_cf_angle', ParameterType.Angle, CutoffType.B_Angle), ('theta_eq', ParameterType.Angle, CutoffType.B_Angle),
            ('j_cf_pauli', ParameterType.Pair, CutoffType.B_Bond)
        ]
    
    @property
    def outputs(self):
        return []

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        flux_charges = torch.zeros(system.topology.natoms, device=system.device, dtype=system.coords.dtype)
        flux_charges_pauli = torch.zeros(system.topology.natoms, device=system.device, dtype=system.coords.dtype)
        if system.topology.angle_atoms.numel() > 0:
            bonded_pair_indices = system.neighbor_list.get_pair_indices(system.topology.bonded_atoms.T)
            angle_pair_indices_ij = system.neighbor_list.get_pair_indices(system.topology.angle_atoms[:, 0:2])
            angle_pair_indices_jk = system.neighbor_list.get_pair_indices(system.topology.angle_atoms[:, 1:].flip(1))
            angles = computeAngleFromVecs(distance_vecs[angle_pair_indices_ij], distance_vecs[angle_pair_indices_jk])
            dists_bonded = dists[bonded_pair_indices]
            dists_angle_ij = dists[angle_pair_indices_ij]
            dists_angle_jk = dists[angle_pair_indices_jk]

            # Bond charge flux params #
            j_cf = system.parameterizer.get_pair_parameters('j_cf', bonded_pair_indices)
            r_eq = system.parameterizer.get_pair_parameters('r_eq', bonded_pair_indices)
            j_cf_pauli = system.parameterizer.get_pair_parameters('j_cf_pauli', bonded_pair_indices)

            # Bond-Bond charge flux params #
            r_eq_bb_1 = system.parameterizer.get_pair_parameters('r_eq', angle_pair_indices_ij)
            r_eq_bb_2 = system.parameterizer.get_pair_parameters('r_eq', angle_pair_indices_jk)
            j_cf_bb_1 = system.parameterizer.get_pair_pair_parameters('j_cf_bb', angle_pair_indices_ij, angle_pair_indices_jk)
            j_cf_bb_2 = system.parameterizer.get_pair_pair_parameters('j_cf_bb', angle_pair_indices_jk, angle_pair_indices_ij)
            
            # Angle charge flux params #
            theta_eq = system.parameterizer.get_angle_parameters('theta_eq', system.topology.angle_atoms)
            j_cf_angle = system.parameterizer.get_angle_parameters('j_cf_angle', system.topology.angle_atoms)

            # bond charge flux #
            charge_flux_bond_1, charge_flux_bond_2 = computeChargeFluxBond(dists_bonded, r_eq, j_cf)
            charge_flux_pauli_bond_1, charge_flux_pauli_bond_2 = computeChargeFluxBond(dists_bonded, r_eq, j_cf_pauli)

            # bond-bond charge flux #
            charge_flux_bb_1, charge_flux_bb_2, charge_flux_bb_3, charge_flux_bb_4 = computeChargeFluxBondBond(
                dists_angle_ij, dists_angle_jk,
                r_eq_bb_1, r_eq_bb_2, j_cf_bb_1, j_cf_bb_2
            )

            # angle charge flux #
            charge_flux_angle_list_i, charge_flux_angle_list_j, charge_flux_angle_list_k = computeChargeFluxAngle(angles, theta_eq, j_cf_angle)

            # Scatter Pauli flux charges to appropriate atom indices #
            flux_charges_pauli.scatter_add_(0, system.topology.bonded_atoms[0], charge_flux_pauli_bond_1)
            flux_charges_pauli.scatter_add_(0, system.topology.bonded_atoms[1], charge_flux_pauli_bond_2)

            # Scatter flux charges to appropriate atom indices #
            flux_charges.scatter_add_(0, system.topology.bonded_atoms[0], charge_flux_bond_1)
            flux_charges.scatter_add_(0, system.topology.bonded_atoms[1], charge_flux_bond_2)
            flux_charges.scatter_add_(0, system.topology.angle_atoms.t()[0], charge_flux_bb_1)
            flux_charges.scatter_add_(0, system.topology.angle_atoms.t()[1], charge_flux_bb_2)
            flux_charges.scatter_add_(0, system.topology.angle_atoms.t()[2], charge_flux_bb_3)
            flux_charges.scatter_add_(0, system.topology.angle_atoms.t()[1], charge_flux_bb_4)
            # ^^^ @SPEED: If it is actually faster, these could be stacked and scattered all at
            # once but I am guessing that is not more efficient.

            flux_charges.scatter_add_(0, system.topology.angle_atoms.T[0], charge_flux_angle_list_i)
            flux_charges.scatter_add_(0, system.topology.angle_atoms.T[1], charge_flux_angle_list_j)
            flux_charges.scatter_add_(0, system.topology.angle_atoms.T[2], charge_flux_angle_list_k)
            
        # Store the flux charges for later use
        system.storage.add('q_flux', flux_charges)
        system.storage.add('q_flux_pauli', flux_charges_pauli)

        return {}