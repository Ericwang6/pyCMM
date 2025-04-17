import torch
from dataclasses import dataclass

from .neighbor_list import *
from typing import Dict, Tuple, List, Optional
from .pbc import applyPBC
from .axis_types import AxisTypes
from .topology import Topology

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False,
           match_args=True, kw_only=True, slots=False, weakref_slot=False)
class Settings:
    pass

class System:
    def __init__(self, coords: torch.Tensor, box: torch.Tensor, cutoff: float, labels: List[str], max_neighbors: int = 1024) -> None:
        pass