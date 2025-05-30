import torch
from ..term import Term
from ...system import System
from ...switching_functions import switch_543

class StoreSwitchingValues(Term):
    def __init__(self):
        super().__init__()
    
    @property
    def param_data(self):
        return []
    
    @property
    def outputs(self):
        return []

    def forward(self, pairs: torch.Tensor, dists: torch.Tensor, distance_vecs: torch.Tensor, system: System):
        settings_short = system.settings.get("short_range")
        settings_long = system.settings.get_long_range_dispersion_settings()
        
        dists_long = system.storage.get('dists_long')
        dists_short = system.storage.get('dists_short')
        #dists_medium = system.storage.get('dists_medium')

        switch_start_short = settings_short.cutoff - settings_short.switching_start_before_cutoff
        switch_start_short = switch_start_short if switch_start_short > 0.0 else 0.0
        switch_short = switch_543(dists_short, switch_start_short, settings_short.cutoff)

        switch_start_long = system.neighbor_list.cutoff - settings_long.switching_start_before_cutoff
        switch_start_long = switch_start_long if switch_start_long > 0.0 else 0.0
        switch_long = switch_543(dists_long, switch_start_long, system.neighbor_list.cutoff)

        system.storage.add('switch_short', switch_short)
        system.storage.add('switch_long', switch_long)

        return {}