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
        self.parameter_metadata = []

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
        # NOTE(JOE): Currently we don't do anything with this metadata.
        # In the future, the point is that we can fill out all of the parameter
        # arrays before evaluating any of the actual terms using this data.
        # I still haven't decided exactly how to do this,
        # but it should make it possible for the loop over terms
        # to be completely static and therefore interoperable with CUDA graphs.
        # It also prevents multiple terms having to generate the same data more than once.
        for param_data in term.param_data:
            self.parameter_metadata.append(param_data)

    @abstractmethod
    def forward(self, system: System):
        raise NotImplementedError