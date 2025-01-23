from ase import Atoms
from ase.calculators.calculator import BaseCalculator

class CMM_ASE(BaseCalculator):
    def __init__(self, parameters=None, use_cache=True):
        super().__init__(parameters, use_cache)