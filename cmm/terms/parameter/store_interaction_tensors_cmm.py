import torch
from ..term import Term
from ...system import System
from ...multipole import computeUndampedInteractionTensorBlocks, formDampingFactorBlocksRank1, formDampingFactorBlocksRank2
from ...short_range import computeShortRangeOneCenterDampFactors, computeShortRangeTwoCenterDampFactors, computeShortRangePolarizationDampFactors
from ...electrostatics import computeDampFactorsErfc, computeDampFactorsErf

class StoreInteractionTensorsCMM(Term):
    def __init__(self):
        super().__init__()
    
    @property
    def param_data(self):
        return []
    
    @property
    def outputs(self):
        return []

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        lr_elec_settings = system.settings.get_long_range_electrostatics_settings()
        
        included_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.included_pairs)
        if system.neighbor_list.excluded_pairs.numel() > 0:
            excluded_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.excluded_pairs)
        else:
            excluded_pair_indices = torch.empty(0, device=included_pair_indices.device, dtype=included_pair_indices.dtype)
        if system.topology.bonded_atoms.numel() > 0:
            bonded_pair_indices = system.neighbor_list.get_pair_indices(system.topology.bonded_atoms.T)
        else:
            bonded_pair_indices = torch.empty(0, device=included_pair_indices.device, dtype=included_pair_indices.dtype)
        if system.topology.angle_atoms.numel() > 0:
            angle_pair_indices_ij = system.neighbor_list.get_pair_indices(system.topology.angle_atoms[:, 0:2])
            angle_pair_indices_jk = system.neighbor_list.get_pair_indices(system.topology.angle_atoms[:, 1:].flip(1))
        else:
            angle_pair_indices_ij = torch.empty(0, device=included_pair_indices.device, dtype=included_pair_indices.dtype)
            angle_pair_indices_jk = torch.empty(0, device=included_pair_indices.device, dtype=included_pair_indices.dtype)

        # Get pairs, dists, and vectors for exclusion list (needed to remove their contribution from long-range interactions) #
        pairs_excl = pairs[excluded_pair_indices, :]
        pairs_excl_i_a = pairs_excl[:, 0]
        pairs_excl_j_a = pairs_excl[:, 1]
        dists_excl = dists[excluded_pair_indices]
        distance_vecs_excl = distance_vecs[excluded_pair_indices]

        # Get pairs, dists, and vectors for vdw potential #
        pairs_vdw = pairs[included_pair_indices, :]
        pairs_vdw_i_a = pairs_vdw[:, 0]
        pairs_vdw_j_a = pairs_vdw[:, 1]
        dists_vdw = dists[included_pair_indices]
        distance_vecs_vdw = distance_vecs[included_pair_indices]

        # Get pairs, dists, and vectors for long-range nonbonded potential #
        indices_vdw_to_lr = torch.where(dists_vdw <= lr_elec_settings.cutoff, torch.arange(dists_vdw.size(0), dtype=torch.long, device=dists_vdw.device), torch.tensor(-1, dtype=torch.long, device=dists_vdw.device))
        indices_vdw_to_lr = indices_vdw_to_lr[indices_vdw_to_lr >= 0]

        pairs_lr = pairs_vdw[indices_vdw_to_lr, :]
        pairs_lr_i_a = pairs_lr[:, 0]
        pairs_lr_j_a = pairs_lr[:, 1]
        dists_lr = dists_vdw[indices_vdw_to_lr]
        distance_vecs_lr = distance_vecs_vdw[indices_vdw_to_lr]

        # Get pairs, dists, and vectors for short-range nonbonded potential #
        indices_vdw_to_sr = torch.where(dists_vdw <= system.settings.get("short_range").cutoff, torch.arange(dists_vdw.size(0), dtype=torch.long, device=dists_vdw.device), torch.tensor(-1, dtype=torch.long, device=dists_vdw.device))
        indices_vdw_to_sr = indices_vdw_to_sr[indices_vdw_to_sr >= 0]
        all_intermolecular_pairs_sr = included_pair_indices[indices_vdw_to_sr]

        pairs_sr = pairs_vdw[indices_vdw_to_sr, :]
        pairs_sr_i_a = pairs_sr[:, 0]
        pairs_sr_j_a = pairs_sr[:, 1]
        dists_sr = dists_vdw[indices_vdw_to_sr]
        distance_vecs_sr = distance_vecs_vdw[indices_vdw_to_sr]

        b_elec = system.parameterizer.get_atomic_parameters('b_elec')
        b_ij_cp_sr_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'b_elec', all_intermolecular_pairs_sr, pairs_sr
        )
        b_ij_pauli_sr_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'b_pauli', all_intermolecular_pairs_sr, pairs_sr
        )
        b_ij_xpol_sr_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'b_xpol', all_intermolecular_pairs_sr, pairs_sr
        )
        b_ij_ct_sr_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'b_ct', all_intermolecular_pairs_sr, pairs_sr
        )

        b_i_elec_p = b_elec[pairs_sr_i_a]
        b_j_elec_p = b_elec[pairs_sr_j_a]
        cp_damps_sr_1c_i = -computeShortRangeOneCenterDampFactors(dists_sr, b_i_elec_p)
        cp_damps_sr_1c_j = -computeShortRangeOneCenterDampFactors(dists_sr, b_j_elec_p)
        cp_damps_sr_2c = -computeShortRangeTwoCenterDampFactors(dists_sr, b_ij_cp_sr_p)
        pauli_damps_sr_2c = computeShortRangeTwoCenterDampFactors(dists_sr, b_ij_pauli_sr_p)
        xpol_damps_sr_2c = -computeShortRangeTwoCenterDampFactors(dists_sr, b_ij_xpol_sr_p)
        ct_damps_sr_2c = -computeShortRangeTwoCenterDampFactors(dists_sr, b_ij_ct_sr_p)
        pol_damps_sr_2c = -computeShortRangePolarizationDampFactors(dists_sr, b_ij_cp_sr_p)
        if lr_elec_settings.use_long_range:
            erfc_damps = computeDampFactorsErfc(dists_lr, lr_elec_settings.alpha) # direct space
            erf_damps = -computeDampFactorsErf(dists_excl, lr_elec_settings.alpha)
            # ^^^ for removing excluded interactions that are implicitly included in long-range summation
            # The reciprocal space calculation uses an erf(alpha*r) damping so the above is -erf(alpha*r)
        else:
            # This is hard-coded to 5 since we always compute damping factors up to quad-quad interactions.
            # In the (distant) future, we should enable automatic detection of multipole rank and try
            # to dispatch batches to kernels which consider the smallest maximum rank allowable. In that
            # case, this 5 would not be hard-coded. The code is going to be so different at that point this
            # comment is hardly worth writing, but at least now you know why there is a 5 here.
            erfc_damps = torch.ones((5, dists_lr.size(0)), device=dists_lr.device)
            erf_damps = torch.zeros((5, dists_excl.size(0)), device=dists_excl.device)

        # Get all undamped and damped interactions needed for multipolar interactions #
        # @SPEED: There are a lot of overlapping calculations here which can be avoided by pulling
        # the interactions tensors out of the long-range one.
        undamped_tensor_1_lr, undamped_tensor_2_lr, undamped_tensor_3_lr = computeUndampedInteractionTensorBlocks(distance_vecs_lr, dists_lr)
        undamped_tensor_1_sr, undamped_tensor_2_sr, undamped_tensor_3_sr = computeUndampedInteractionTensorBlocks(distance_vecs_sr, dists_sr)
        undamped_tensor_1_excl, undamped_tensor_2_excl, undamped_tensor_3_excl = computeUndampedInteractionTensorBlocks(distance_vecs_excl, dists_excl)
        undamped_tensor_1_pol_sr = undamped_tensor_1_sr[:, :4, :4]
        undamped_tensor_2_pol_sr = undamped_tensor_2_sr[:, :4, :4]

        ewald_damps_lr_1, ewald_damps_lr_2, ewald_damps_lr_3 = formDampingFactorBlocksRank2(erfc_damps)
        ewald_damps_excl_1, ewald_damps_excl_2, ewald_damps_excl_3 = formDampingFactorBlocksRank2(erf_damps)
        cp_damps_sr_1c_1_i, cp_damps_sr_1c_2_i, cp_damps_sr_1c_3_i = formDampingFactorBlocksRank2(cp_damps_sr_1c_i)
        cp_damps_sr_1c_1_j, cp_damps_sr_1c_2_j, cp_damps_sr_1c_3_j = formDampingFactorBlocksRank2(cp_damps_sr_1c_j)
        cp_damps_sr_2c_1, cp_damps_sr_2c_2, cp_damps_sr_2c_3 = formDampingFactorBlocksRank2(cp_damps_sr_2c)
        pauli_damps_sr_2c_1, pauli_damps_sr_2c_2, pauli_damps_sr_2c_3 = formDampingFactorBlocksRank2(pauli_damps_sr_2c)
        xpol_damps_sr_2c_1, xpol_damps_sr_2c_2, xpol_damps_sr_2c_3 = formDampingFactorBlocksRank2(xpol_damps_sr_2c)
        ct_damps_sr_2c_1, ct_damps_sr_2c_2, ct_damps_sr_2c_3 = formDampingFactorBlocksRank2(ct_damps_sr_2c)
        pol_damps_sr_2c_1, pol_damps_sr_2c_2 = formDampingFactorBlocksRank1(pol_damps_sr_2c)

        direct_field_tensor_lr = torch.mul(undamped_tensor_1_lr, ewald_damps_lr_1) + torch.mul(undamped_tensor_2_lr, ewald_damps_lr_2) + torch.mul(undamped_tensor_3_lr, ewald_damps_lr_3)
        direct_field_tensor_excl = torch.mul(undamped_tensor_1_excl, ewald_damps_excl_1) + torch.mul(undamped_tensor_2_excl, ewald_damps_excl_2) + torch.mul(undamped_tensor_3_excl, ewald_damps_excl_3)
        direct_field_tensor_rank_1_lr = direct_field_tensor_lr[:, :4, :4]
        direct_field_tensor_excl_rank_1 = direct_field_tensor_excl[:, :4, :4]
        # ^^^^ Gets just the entries needed for charges and dipoles (for polarization)
        
        cp_field_tensor_sr_i = torch.mul(undamped_tensor_1_sr, cp_damps_sr_1c_1_i) + torch.mul(undamped_tensor_2_sr, cp_damps_sr_1c_2_i) + torch.mul(undamped_tensor_3_sr, cp_damps_sr_1c_3_i)
        cp_field_tensor_sr_j = torch.mul(undamped_tensor_1_sr, cp_damps_sr_1c_1_j) + torch.mul(undamped_tensor_2_sr, cp_damps_sr_1c_2_j) + torch.mul(undamped_tensor_3_sr, cp_damps_sr_1c_3_j)
        cp_interaction_tensor_sr = torch.mul(undamped_tensor_1_sr, cp_damps_sr_2c_1) + torch.mul(undamped_tensor_2_sr, cp_damps_sr_2c_2) + torch.mul(undamped_tensor_3_sr, cp_damps_sr_2c_3)
        pauli_interaction_tensor_sr = torch.mul(undamped_tensor_1_sr, pauli_damps_sr_2c_1) + torch.mul(undamped_tensor_2_sr, pauli_damps_sr_2c_2) + torch.mul(undamped_tensor_3_sr, pauli_damps_sr_2c_3)
        xpol_interaction_tensor_sr = torch.mul(undamped_tensor_1_sr, xpol_damps_sr_2c_1) + torch.mul(undamped_tensor_2_sr, xpol_damps_sr_2c_2) + torch.mul(undamped_tensor_3_sr, xpol_damps_sr_2c_3)
        ct_interaction_tensor_sr = torch.mul(undamped_tensor_1_sr, ct_damps_sr_2c_1) + torch.mul(undamped_tensor_2_sr, ct_damps_sr_2c_2) + torch.mul(undamped_tensor_3_sr, ct_damps_sr_2c_3)
        pol_interaction_tensor_sr = torch.mul(undamped_tensor_1_pol_sr, pol_damps_sr_2c_1) + torch.mul(undamped_tensor_2_pol_sr, pol_damps_sr_2c_2)

        system.storage.add('direct_field_tensor_lr', direct_field_tensor_lr)
        system.storage.add('direct_field_tensor_excl', direct_field_tensor_excl)
        system.storage.add('direct_field_tensor_rank_1_lr', direct_field_tensor_rank_1_lr)
        system.storage.add('direct_field_tensor_excl_rank_1', direct_field_tensor_excl_rank_1)
        system.storage.add('cp_field_tensor_sr_i', cp_field_tensor_sr_i)
        system.storage.add('cp_field_tensor_sr_j', cp_field_tensor_sr_j)
        system.storage.add('cp_interaction_tensor_sr', cp_interaction_tensor_sr)
        system.storage.add('pauli_interaction_tensor_sr', pauli_interaction_tensor_sr)
        system.storage.add('xpol_interaction_tensor_sr', xpol_interaction_tensor_sr)
        system.storage.add('ct_interaction_tensor_sr', ct_interaction_tensor_sr)
        system.storage.add('pol_interaction_tensor_sr', pol_interaction_tensor_sr)

        return {}