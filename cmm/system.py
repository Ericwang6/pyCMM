import torch
from iodata import load_one
from .neighbor_list import *
from typing import Dict, Tuple, List, Optional
from .pbc import applyPBC
from .axis_types import AxisTypes
from .topology import Topology

def create_system_from_ext_xyz_file(file_name: str, requires_grad: bool=True, device: str="cpu"):
    mol = load_one(file_name, fmt="extxyz")
    print(mol)
    

class System:
    def __init__(self) -> None:
        pass