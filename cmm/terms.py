import torch
from .bonded import *

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

def evaluate_hardness_change(
    pairs: torch.Tensor, dists: torch.Tensor, angles: torch.Tensor,
    bonded_pairs: torch.Tensor, angle_pairs: torch.Tensor, angle_atoms: torch.Tensor,
    eta: torch.Tensor, r_eq: torch.Tensor, theta_eq: torch.Tensor,
    k_hardness_b: torch.Tensor,  k_hardness_angle: torch.Tensor,
    r_eq_bb_1: torch.Tensor, r_eq_bb_2: torch.Tensor,
    k_hardness_bb_1: torch.Tensor, k_hardness_bb_2: torch.Tensor):
    
    hardness_product = torch.ones_like(eta)
    hardness_change_b = computeHardnessChangeBond(dists[bonded_pairs], r_eq, k_hardness_b)
    hardness_change_bb_1, hardness_change_bb_2 = computeHardnessChangeBondBond(
        dists[angle_pairs[0]], dists[angle_pairs[1]],
        r_eq_bb_1, r_eq_bb_2, k_hardness_bb_1, k_hardness_bb_2
    )
    hardness_change_angle = computeHardnessChangeAngle(angles, theta_eq, k_hardness_angle)
        
    # NOTE(JOE): We only accumulate some of the bond-bond terms since this basically assumes that all of
    # the hardness change is on the second atom of the bond. i.e. just the H atoms in water.
    # This is yet another reason not to like this piece of the model. There should be a better way.

    # Cannot do in-place operations or else computational graphs breaks. Sad.
    bonded_atoms = pairs[bonded_pairs].T
    hardness_product = hardness_product.scatter_reduce(0, bonded_atoms[1], hardness_change_b, reduce="prod")
    hardness_product = hardness_product.scatter_reduce(0, pairs[angle_pairs[0]].T[1], hardness_change_bb_1, reduce="prod")
    hardness_product = hardness_product.scatter_reduce(0, pairs[angle_pairs[1]].T[1], hardness_change_bb_2, reduce="prod")

    eta *= hardness_product
    eta.scatter_add_(0, angle_atoms.T[0], hardness_change_angle)
    eta.scatter_add_(0, angle_atoms.T[2], hardness_change_angle)