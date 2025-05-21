import torch
from abc import ABC, abstractmethod
from enum import IntEnum
from ..system import System

class CutoffType(IntEnum):
    NB_Short   = 0
    NB_Medium  = 1
    NB_Long    = 2
    NB_Ewald   = 3
    B_Bond     = 4
    B_Angle    = 5
    B_Dihedral = 6

class ParameterType(IntEnum):
    Atomic    = 0
    Pair      = 1
    Angle     = 2
    Dihedral  = 3
    PairPair  = 4
    PairAngle = 5
    # ...

class Term(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def cutoff_type(self):
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