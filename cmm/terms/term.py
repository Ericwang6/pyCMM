import torch
from abc import ABC, abstractmethod
from enum import IntEnum
from ..system import System

class InteractionType(IntEnum):
    Bonded         = 0
    Electrostatics = 1
    Pauli          = 2
    Dispersion     = 3
    Polarization   = 4
    ChargeTransfer = 5
    Parameter      = 6
    Other          = 7

class Term(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def interaction_type(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def outputs(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, system: System):
        raise NotImplementedError