import torch
from ..system import System
from ..parameters import Parameterizer
from ..terms.term import Term
from abc import ABC, abstractmethod

class FF(torch.nn.Module, ABC):
    def __init__(self, dtype: torch.dtype=torch.float64, device: torch.DeviceObjType=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device
        self._ff_parameters = {}
        self._terms = []
        self._energies = {}

    def add_term(self, term: Term):
        self._terms.append(term)
        for name in term.params:
            self._ff_parameters[name] = torch.empty(0, dtype=self._dtype, device=self._device)

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