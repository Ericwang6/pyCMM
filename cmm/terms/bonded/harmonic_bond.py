import torch
from ..term import Term, InteractionType
from ...system import System

class HarmonicBond(Term):
    @property
    def interaction_type(self):
        return InteractionType.Bonded
    
    @property
    def params(self):
        return ['k_bond', 'r_eq']
    
    @property
    def outputs(self):
        return ['V_bond']

    def forward(self, system: System):
        print(self.parameters())
        #system.parameterizer.get_params(self.parameters())
    

