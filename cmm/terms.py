import torch
from .coordinate_manager import CoordinateManager
from .bonded import *

# This file provides convenient wrappers of combinations of computational kernels
# which might be used by a force field. It is also possible to define these wrapper
# methods inside of a force field as needed.

def evaluate_bond_charge_flux(
    pairs: torch.Tensor, dists: torch.Tensor, bonded_pairs: torch.Tensor,
    q: torch.Tensor, r_eq: torch.Tensor, j_cf: torch.Tensor
):
    charge_flux_bond_1, charge_flux_bond_2 = computeChargeFluxBond(dists[bonded_pairs], r_eq, j_cf)
    bonded_pairs = pairs[bonded_pairs].T
    flux_charges = torch.zeros_like(q)
    flux_charges.scatter_add_(0, bonded_pairs[0], charge_flux_bond_1)
    flux_charges.scatter_add_(0, bonded_pairs[1], charge_flux_bond_2)
    q.add_(flux_charges)

def evaluate_bond_and_angle_charge_flux(
        pairs: torch.Tensor, dists: torch.Tensor, angles: torch.Tensor,
        bonded_pairs: torch.Tensor, angle_pairs: torch.Tensor, angle_atoms: torch.Tensor,
        q: torch.Tensor, r_eq: torch.Tensor, theta_eq: torch.Tensor,
        j_cf: torch.Tensor,  j_cf_angle: torch.Tensor, r_eq_bb_1: torch.Tensor, r_eq_bb_2: torch.Tensor,
        j_cf_bb_1: torch.Tensor, j_cf_bb_2: torch.Tensor):

    # bond charge flux #
    charge_flux_bond_1, charge_flux_bond_2 = computeChargeFluxBond(dists[bonded_pairs], r_eq, j_cf)

    # bond-bond coupling #
    charge_flux_bb_1, charge_flux_bb_2, charge_flux_bb_3, charge_flux_bb_4 = computeChargeFluxBondBond(
        dists[angle_pairs[0]], dists[angle_pairs[1]],
        r_eq_bb_1, r_eq_bb_2, j_cf_bb_1, j_cf_bb_2
    )

    # angle charge flux #
    charge_flux_angle_list_i, charge_flux_angle_list_j, charge_flux_angle_list_k = computeChargeFluxAngle(angles, theta_eq, j_cf_angle)

    # Scatter flux charges to appropriate atom indices #
    bonded_atoms = pairs[bonded_pairs].T
    flux_charges = torch.zeros_like(q) # @SPEED: Ideally we could do this without allocating.
    flux_charges.scatter_add_(0, bonded_atoms[0], charge_flux_bond_1)
    flux_charges.scatter_add_(0, bonded_atoms[1], charge_flux_bond_2)

    flux_charges.scatter_add_(0, pairs[angle_pairs[0]].T[0], charge_flux_bb_1)
    flux_charges.scatter_add_(0, pairs[angle_pairs[0]].T[1], charge_flux_bb_2)
    flux_charges.scatter_add_(0, pairs[angle_pairs[1]].T[0], charge_flux_bb_3)
    flux_charges.scatter_add_(0, pairs[angle_pairs[1]].T[1], charge_flux_bb_4)
    # ^^^ @SPEED: If it is actually faster, these could be stacked and scattered all at
    # once but I am guessing that is not more efficient.

    flux_charges.scatter_add_(0, angle_atoms[:, 0], charge_flux_angle_list_i)
    flux_charges.scatter_add_(0, angle_atoms[:, 1], charge_flux_angle_list_j)
    flux_charges.scatter_add_(0, angle_atoms[:, 2], charge_flux_angle_list_k)
    q.add_(flux_charges)