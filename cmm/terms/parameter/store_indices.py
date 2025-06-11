import torch
from ..term import Term
from ...system import System

class StoreIndices(Term):
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
        
        included_pair_indices_long = system.neighbor_list.get_pair_indices(system.neighbor_list.included_pairs)
        if system.neighbor_list.excluded_pairs.numel() > 0:
            excluded_pair_indices = system.neighbor_list.get_pair_indices(system.neighbor_list.excluded_pairs)
        else:
            excluded_pair_indices = torch.empty(0, device=included_pair_indices_long.device, dtype=included_pair_indices_long.dtype)
        if system.topology.bonded_atoms.numel() > 0:
            bonded_pair_indices = system.neighbor_list.get_pair_indices(system.topology.bonded_atoms.T)
        else:
            bonded_pair_indices = torch.empty(0, device=included_pair_indices_long.device, dtype=included_pair_indices_long.dtype)
        if system.topology.angle_atoms.numel() > 0:
            angle_pair_indices_ij = system.neighbor_list.get_pair_indices(system.topology.angle_atoms[:, 0:2])
            angle_pair_indices_jk = system.neighbor_list.get_pair_indices(system.topology.angle_atoms[:, 1:].flip(1))
        else:
            angle_pair_indices_ij = torch.empty(0, device=included_pair_indices_long.device, dtype=included_pair_indices_long.dtype)
            angle_pair_indices_jk = torch.empty(0, device=included_pair_indices_long.device, dtype=included_pair_indices_long.dtype)

        # Get pairs, dists, and vectors for exclusion list (needed to remove their contribution from long-range interactions) #
        pairs_excl = pairs[excluded_pair_indices, :]
        dists_excl = dists[excluded_pair_indices]
        distance_vecs_excl = distance_vecs[excluded_pair_indices]

        # Get pairs, dists, and vectors for van der waal's potential #
        pairs_long = pairs[included_pair_indices_long, :]
        dists_long = dists[included_pair_indices_long]
        distance_vecs_long = distance_vecs[included_pair_indices_long]

        # Get pairs, dists, and vectors for medium-range nonbonded potential #
        indices_long_to_medium = torch.where(dists_long <= lr_elec_settings.cutoff, torch.arange(dists_long.size(0), dtype=torch.long, device=dists_long.device), torch.tensor(-1, dtype=torch.long, device=dists_long.device))
        indices_long_to_medium = indices_long_to_medium[indices_long_to_medium >= 0]

        included_pair_indices_medium = included_pair_indices_long[indices_long_to_medium]
        pairs_medium = pairs_long[indices_long_to_medium, :]
        dists_medium = dists_long[indices_long_to_medium]
        distance_vecs_medium = distance_vecs_long[indices_long_to_medium]

        # Get pairs, dists, and vectors for short-range nonbonded potential #
        indices_long_to_short = torch.where(dists_long <= system.settings.get("short_range").cutoff, torch.arange(dists_long.size(0), dtype=torch.long, device=dists_long.device), torch.tensor(-1, dtype=torch.long, device=dists_long.device))
        indices_long_to_short = indices_long_to_short[indices_long_to_short >= 0]

        included_pair_indices_short = included_pair_indices_long[indices_long_to_short]
        pairs_short = pairs_long[indices_long_to_short, :]
        dists_short = dists_long[indices_long_to_short]
        distance_vecs_short = distance_vecs_long[indices_long_to_short]

        pairs_pol = torch.cat((pairs_short, system.topology.angle_atoms[:, [0, 2]], system.topology.angle_atoms[:, [2, 0]]))
        included_pair_indices_pol = system.neighbor_list.get_pair_indices(pairs_pol)
        dists_pol = dists[included_pair_indices_pol]
        distance_vecs_pol = distance_vecs[included_pair_indices_pol]

        pairs_medium_pol = torch.cat((pairs_medium, system.topology.angle_atoms[:, [0, 2]], system.topology.angle_atoms[:, [2, 0]]))
        included_pair_indices_medium_pol = system.neighbor_list.get_pair_indices(pairs_medium_pol)
        dists_medium_pol = dists[included_pair_indices_medium_pol]
        distance_vecs_medium_pol = distance_vecs[included_pair_indices_medium_pol]


        pairs_excl_pol = pairs[bonded_pair_indices, :]
        dists_excl_pol = dists[bonded_pair_indices]
        distance_vecs_excl_pol = distance_vecs[bonded_pair_indices]

        system.storage.add('pairs_excl', pairs_excl)
        system.storage.add('dists_excl', dists_excl)
        system.storage.add('distance_vecs_excl', distance_vecs_excl)
        system.storage.add('excluded_pair_indices', excluded_pair_indices)
        
        system.storage.add('pairs_long', pairs_long)
        system.storage.add('dists_long', dists_long)
        system.storage.add('distance_vecs_long', distance_vecs_long)
        system.storage.add('included_pair_indices_long', included_pair_indices_long)
        
        system.storage.add('pairs_medium', pairs_medium)
        system.storage.add('dists_medium', dists_medium)
        system.storage.add('distance_vecs_medium', distance_vecs_medium)
        system.storage.add('included_pair_indices_medium', included_pair_indices_medium)
        
        system.storage.add('pairs_short', pairs_short)
        system.storage.add('dists_short', dists_short)
        system.storage.add('distance_vecs_short', distance_vecs_short)
        system.storage.add('included_pair_indices_short', included_pair_indices_short)
        
        system.storage.add('pairs_excl_pol', pairs_excl_pol)
        system.storage.add('dists_excl_pol', dists_excl_pol)
        system.storage.add('distance_vecs_excl_pol', distance_vecs_excl_pol)
        # NOTE(JOE): The excluded pair indices for polarization are exactly the
        # bonded pair indices since we include 1-3 electrostatic interactions between
        # induced multipoles.

        system.storage.add('pairs_pol', pairs_pol)
        system.storage.add('dists_pol', dists_pol)
        system.storage.add('distance_vecs_pol', distance_vecs_pol)
        system.storage.add('included_pair_indices_pol', included_pair_indices_pol)

        system.storage.add('pairs_medium_pol', pairs_medium_pol)
        system.storage.add('included_pair_indices_medium_pol', included_pair_indices_medium_pol)
        system.storage.add('dists_medium_pol', dists_medium_pol)
        system.storage.add('distance_vecs_medium_pol', distance_vecs_medium_pol)
        
        system.storage.add('bonded_pair_indices', bonded_pair_indices)
        system.storage.add('angle_pair_indices_ij', angle_pair_indices_ij)
        system.storage.add('angle_pair_indices_jk', angle_pair_indices_jk)

        return {}