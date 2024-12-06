import torch
from .coordinate_manager import CoordinateManager
from .bonded import *

# This file provides convenient wrappers of combinations of computational kernels
# which might be used by a force field. It is also possible to define these wrapper
# methods inside of a force field as needed.

def evaluate_bond_charge_flux(
    bond_dists: torch.Tensor, bond_indices: torch.Tensor,
    q: torch.Tensor, r_eq: torch.Tensor, j_cf: torch.Tensor
):
    charge_flux_bond_1, charge_flux_bond_2 = computeChargeFluxBond(bond_dists, r_eq, j_cf)
    
    flux_charges = torch.zeros_like(q)
    flux_charges.scatter_add_(0, bond_indices[0], charge_flux_bond_1)
    flux_charges.scatter_add_(0, bond_indices[1], charge_flux_bond_2)
    q.add_(flux_charges)

def evaluate_bond_and_angle_charge_flux(
        bond_dists: torch.Tensor, angles: torch.Tensor,
        bond_indices: torch.Tensor, bond_bond_indices: torch.Tensor, angle_indices: torch.Tensor,
        q: torch.Tensor, r_eq: torch.Tensor, theta_eq: torch.Tensor,
        j_cf: torch.Tensor, j_cf_bb: torch.Tensor, j_cf_angle: torch.Tensor):

    # bond charge flux #
    charge_flux_bond_1, charge_flux_bond_2 = computeChargeFluxBond(bond_dists, r_eq, j_cf)

    # bond-bond coupling #
    charge_flux_bb_1, charge_flux_bb_2, charge_flux_bb_3, charge_flux_bb_4 = computeChargeFluxBondBond(
        bond_dists[bond_bond_indices[0]], bond_dists[bond_bond_indices[1]],
        r_eq[bond_bond_indices[0]], r_eq[bond_bond_indices[1]],
        j_cf_bb[bond_bond_indices[0]], j_cf_bb[bond_bond_indices[1]]
    )

    # angle charge flux #
    charge_flux_angle_list_i, charge_flux_angle_list_j, charge_flux_angle_list_k = computeChargeFluxAngle(angles, theta_eq, j_cf_angle)

    # Scatter flux charges to appropriate indices of array #
    flux_charges = torch.zeros_like(q) # @SPEED: Ideally we could do this without allocating.
    
    flux_charges.scatter_add_(0, bond_indices[0], charge_flux_bond_1)
    flux_charges.scatter_add_(0, bond_indices[1], charge_flux_bond_2)

    flux_charges.scatter_add_(0, bond_indices.T[bond_bond_indices[0]].T[0], charge_flux_bb_1)
    flux_charges.scatter_add_(0, bond_indices.T[bond_bond_indices[0]].T[1], charge_flux_bb_2)
    flux_charges.scatter_add_(0, bond_indices.T[bond_bond_indices[1]].T[0], charge_flux_bb_3)
    flux_charges.scatter_add_(0, bond_indices.T[bond_bond_indices[1]].T[1], charge_flux_bb_4)
    # ^^^ @SPEED: If it is actually faster, these could be stacked and scattered all at
    # once but I am guessing that is not more efficient.

    flux_charges.scatter_add_(0, angle_indices[0], charge_flux_angle_list_i)
    flux_charges.scatter_add_(0, angle_indices[1], charge_flux_angle_list_j)
    flux_charges.scatter_add_(0, angle_indices[2], charge_flux_angle_list_k)
    q.add_(flux_charges)