import torch
from ..system import System
from ..parameters import Parameterizer
from ..terms.term import Term
from ..terms.bonded import *
from ..terms.nonbonded import *
from ..settings import Settings
from abc import ABC, abstractmethod

class FF(torch.nn.Module, ABC):
    def __init__(self, system: System, dtype: torch.dtype=torch.float64, device: torch.DeviceObjType=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device

        self.terms = []
        self.energies = {}

    def setup_long_range_interactions(self, system: System):
        lr_elec_settings = system.settings.get_long_range_electrostatics_settings()
        lr_disp_settings = system.settings.get_long_range_dispersion_settings()
        if lr_elec_settings.use_long_range:
            if lr_elec_settings.method == "ewald":
                self.add_term(EwaldEnergy0(lr_elec_settings.alpha, lr_elec_settings.k_max))
                # ^^^ Eventually change to n_x, n_y, n_z and store them on the Term instead of k_max.
        if lr_disp_settings.use_long_range:
            if lr_disp_settings.method == "lrc":
                self.add_term(LongRangeLennardJonesCorrection())
                # ^^^ Eventually change to n_x, n_y, n_z and store them on the Term instead of k_max.

    def add_term(self, term: Term):
        self.terms.append(term)

    @abstractmethod
    def forward(self, system: System):
        # 1) Fill out the parameters
        # 2) Evaluate any dependencies
        # 3) Update parameters which need to be updated (possibly with a different name to prevent overwriting otherwise constant params?)
        # 4) Evaluate remaining terms, continuing the process if needed.
        
        # NOTE(JOE): Conceivably, there can be a loop in here where we evaluate some function (like the polarization)
        # then update the parameters and repeat this until convergence. That is a bit of an edge case but it should be possible.
        # I think we can just make this a special type of term which updates both parameters and an energy.
        raise NotImplementedError