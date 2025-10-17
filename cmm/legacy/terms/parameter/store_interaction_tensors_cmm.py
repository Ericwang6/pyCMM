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

        dists_excl = system.storage.get('dists_excl')
        distance_vecs_excl = system.storage.get('distance_vecs_excl')
        dists_medium = system.storage.get('dists_medium')
        distance_vecs_medium = system.storage.get('distance_vecs_medium')
        included_pair_indices_short = system.storage.get('included_pair_indices_short')
        pairs_short = system.storage.get('pairs_short')
        dists_short = system.storage.get('dists_short')
        distance_vecs_short = system.storage.get('distance_vecs_short')

        b_elec = system.parameterizer.get_atomic_parameters('b_elec')
        b_ij_cp_short_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'b_elec', included_pair_indices_short, pairs_short
        )
        b_ij_pauli_short_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'b_pauli', included_pair_indices_short, pairs_short
        )
        b_ij_xpol_short_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'b_xpol', included_pair_indices_short, pairs_short
        )
        b_ij_ct_short_p = system.parameterizer.get_pair_parameters_with_optional_combination_rule(
            'b_ct', included_pair_indices_short, pairs_short
        )

        b_i_elec_p = b_elec[pairs_short[:, 0]]
        b_j_elec_p = b_elec[pairs_short[:, 1]]
        cp_damps_short_1c_i = -computeShortRangeOneCenterDampFactors(dists_short, b_i_elec_p)
        cp_damps_short_1c_j = -computeShortRangeOneCenterDampFactors(dists_short, b_j_elec_p)
        cp_damps_short_2c = -computeShortRangeTwoCenterDampFactors(dists_short, b_ij_cp_short_p)
        pauli_damps_short_2c = computeShortRangeTwoCenterDampFactors(dists_short, b_ij_pauli_short_p)
        xpol_damps_short_2c = -computeShortRangeTwoCenterDampFactors(dists_short, b_ij_xpol_short_p)
        ct_damps_short_2c = -computeShortRangeTwoCenterDampFactors(dists_short, b_ij_ct_short_p)
        pol_damps_short_2c = -computeShortRangePolarizationDampFactors(dists_short, b_ij_cp_short_p)
        if lr_elec_settings.use_long_range:
            erfc_damps = computeDampFactorsErfc(dists_medium, lr_elec_settings.alpha) # direct space
            erf_damps = -computeDampFactorsErf(dists_excl, lr_elec_settings.alpha)
            # ^^^ for removing excluded interactions that are implicitly included in medium-range summation
            # The reciprocal space calculation uses an erf(alpha*r) damping so the above is -erf(alpha*r)
        else:
            # This is hard-coded to 5 since we always compute damping factors up to quad-quad interactions.
            # In the (distant) future, we should enable automatic detection of multipole rank and try
            # to dispatch batches to kernels which consider the smallest maximum rank allowable. In that
            # case, this 5 would not be hard-coded. The code is going to be so different at that point this
            # comment is hardly worth writing, but at least now you know why there is a 5 here.
            erfc_damps = torch.ones((5, dists_medium.size(0)), device=dists_medium.device)
            erf_damps = torch.zeros((5, dists_excl.size(0)), device=dists_excl.device)

        # Get all undamped and damped interactions needed for multipolar interactions #
        # @SPEED: There are a lot of overlapping calculations here which can be avoided by pulling
        # the interactions tensors out of the long-range one.
        undamped_tensor_1_medium, undamped_tensor_2_medium, undamped_tensor_3_medium = computeUndampedInteractionTensorBlocks(distance_vecs_medium, dists_medium)
        undamped_tensor_1_short, undamped_tensor_2_short, undamped_tensor_3_short = computeUndampedInteractionTensorBlocks(distance_vecs_short, dists_short)
        undamped_tensor_1_excl, undamped_tensor_2_excl, undamped_tensor_3_excl = computeUndampedInteractionTensorBlocks(distance_vecs_excl, dists_excl)
        undamped_tensor_1_pol_short = undamped_tensor_1_short[:, :4, :4]
        undamped_tensor_2_pol_short = undamped_tensor_2_short[:, :4, :4]

        ewald_damps_medium_1, ewald_damps_medium_2, ewald_damps_medium_3 = formDampingFactorBlocksRank2(erfc_damps)
        ewald_damps_excl_1, ewald_damps_excl_2, ewald_damps_excl_3 = formDampingFactorBlocksRank2(erf_damps)
        cp_damps_short_1c_1_i, cp_damps_short_1c_2_i, cp_damps_short_1c_3_i = formDampingFactorBlocksRank2(cp_damps_short_1c_i)
        cp_damps_short_1c_1_j, cp_damps_short_1c_2_j, cp_damps_short_1c_3_j = formDampingFactorBlocksRank2(cp_damps_short_1c_j)
        cp_damps_short_2c_1, cp_damps_short_2c_2, cp_damps_short_2c_3 = formDampingFactorBlocksRank2(cp_damps_short_2c)
        pauli_damps_short_2c_1, pauli_damps_short_2c_2, pauli_damps_short_2c_3 = formDampingFactorBlocksRank2(pauli_damps_short_2c)
        xpol_damps_short_2c_1, xpol_damps_short_2c_2, xpol_damps_short_2c_3 = formDampingFactorBlocksRank2(xpol_damps_short_2c)
        ct_damps_short_2c_1, ct_damps_short_2c_2, ct_damps_short_2c_3 = formDampingFactorBlocksRank2(ct_damps_short_2c)
        pol_damps_short_2c_1, pol_damps_short_2c_2 = formDampingFactorBlocksRank1(pol_damps_short_2c)

        direct_field_tensor_medium = torch.mul(undamped_tensor_1_medium, ewald_damps_medium_1) + torch.mul(undamped_tensor_2_medium, ewald_damps_medium_2) + torch.mul(undamped_tensor_3_medium, ewald_damps_medium_3)
        direct_field_tensor_excl = torch.mul(undamped_tensor_1_excl, ewald_damps_excl_1) + torch.mul(undamped_tensor_2_excl, ewald_damps_excl_2) + torch.mul(undamped_tensor_3_excl, ewald_damps_excl_3)
        direct_field_tensor_rank_1_medium = direct_field_tensor_medium[:, :4, :4]
        direct_field_tensor_excl_rank_1 = direct_field_tensor_excl[:, :4, :4]
        # ^^^^ Gets just the entries needed for charges and dipoles (for polarization)
        
        cp_field_tensor_short_i = torch.mul(undamped_tensor_1_short, cp_damps_short_1c_1_i) + torch.mul(undamped_tensor_2_short, cp_damps_short_1c_2_i) + torch.mul(undamped_tensor_3_short, cp_damps_short_1c_3_i)
        cp_field_tensor_short_j = torch.mul(undamped_tensor_1_short, cp_damps_short_1c_1_j) + torch.mul(undamped_tensor_2_short, cp_damps_short_1c_2_j) + torch.mul(undamped_tensor_3_short, cp_damps_short_1c_3_j)
        cp_interaction_tensor_short = torch.mul(undamped_tensor_1_short, cp_damps_short_2c_1) + torch.mul(undamped_tensor_2_short, cp_damps_short_2c_2) + torch.mul(undamped_tensor_3_short, cp_damps_short_2c_3)
        pauli_interaction_tensor_short = torch.mul(undamped_tensor_1_short, pauli_damps_short_2c_1) + torch.mul(undamped_tensor_2_short, pauli_damps_short_2c_2) + torch.mul(undamped_tensor_3_short, pauli_damps_short_2c_3)
        xpol_interaction_tensor_short = torch.mul(undamped_tensor_1_short, xpol_damps_short_2c_1) + torch.mul(undamped_tensor_2_short, xpol_damps_short_2c_2) + torch.mul(undamped_tensor_3_short, xpol_damps_short_2c_3)
        ct_interaction_tensor_short = torch.mul(undamped_tensor_1_short, ct_damps_short_2c_1) + torch.mul(undamped_tensor_2_short, ct_damps_short_2c_2) + torch.mul(undamped_tensor_3_short, ct_damps_short_2c_3)
        pol_interaction_tensor_short = torch.mul(undamped_tensor_1_pol_short, pol_damps_short_2c_1) + torch.mul(undamped_tensor_2_pol_short, pol_damps_short_2c_2)

        system.storage.add('direct_field_tensor_medium', direct_field_tensor_medium)
        system.storage.add('direct_field_tensor_excl', direct_field_tensor_excl)
        system.storage.add('direct_field_tensor_rank_1_medium', direct_field_tensor_rank_1_medium)
        system.storage.add('direct_field_tensor_excl_rank_1', direct_field_tensor_excl_rank_1)
        system.storage.add('cp_field_tensor_short_i', cp_field_tensor_short_i)
        system.storage.add('cp_field_tensor_short_j', cp_field_tensor_short_j)
        system.storage.add('cp_interaction_tensor_short', cp_interaction_tensor_short)
        system.storage.add('pauli_interaction_tensor_short', pauli_interaction_tensor_short)
        system.storage.add('xpol_interaction_tensor_short', xpol_interaction_tensor_short)
        system.storage.add('ct_interaction_tensor_short', ct_interaction_tensor_short)
        system.storage.add('pol_interaction_tensor_short', pol_interaction_tensor_short)

        return {}