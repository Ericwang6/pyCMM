import torch
from iodata import load_one
from .neighbor_list import *
from typing import Dict, Tuple, List, Optional
from .pbc import applyPBC
from .axis_types import AxisTypes
from .topology import Topology
from .settings import *
from .parameters import Parameterizer2
from .units import BOHR2ANG

def create_system_from_ext_xyz_file(file_name: str, settings: Settings, requires_grad: bool=True, device: str="cpu"):
    mol = load_one(file_name, fmt="extxyz")
    print(mol)
    
class System:
    def __init__(self, coords: torch.Tensor, box: torch.Tensor, atom_type_names: List[str], topology: Topology, settings: Settings,
                 device: torch.DeviceObjType=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.device = device
        self._need_coordinate_grads = coords.requires_grad
        self._need_box_grads = box.requires_grad
        self.coords = coords
        self.box = box
        self.box_inv = torch.inverse(self.box)
        self.box_lengths = torch.diagonal(self.box)
        self.box_volume = torch.det(self.box)
        self.parameterizer = Parameterizer2(atom_type_names, device=self.device)
        self.topology = topology
        self.settings = settings
        self._check_for_nl_update = False
        self.build_neighbor_list()

    
    def build_neighbor_list(self):
        nl_settings = self.settings.get("neighbor_list")
        if nl_settings.method == "verlet":
            self.neighbor_list = VerletList2(
                self.coords, self.box, nl_settings.cutoff,
                padding=torch.tensor(nl_settings.padding / BOHR2ANG, device=self.coords.device)
            )
        # elif nl_settings.method == "whatever":
        else:
            raise NotImplementedError
    
    def update_coordinates(self, new_coords: torch.Tensor):
        """
        Update coordinates held by the coordinate manager.
        """
        self.coords = new_coords.detach().clone().to(torch.get_default_dtype()).requires_grad_(self._need_coordinate_grads)
        self._check_for_nl_update = True

    def update_box(self, new_box: torch.Tensor):
        """
        Update box vectors and related quantities.
        """
        self.box = new_box.detach().clone().to(torch.get_default_dtype()).requires_grad_(self._need_box_grads)
        
        # Update related quantities
        self.box_inv = torch.inverse(self.box)
        self.box_lengths = torch.diagonal(self.box)
        self.box_volume = torch.det(self.box)
        self._check_for_nl_update = True
        
    def get_distances_vectors_and_pairs(self, reset_grads=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all indices of atom pairs, distances, distance vectors.
        """

        # Only reset gradients if explicitly requested
        if reset_grads: 
            if self._need_coordinate_grads:
                self.update_coordinates(self.coords)
            if self._need_box_grads:
                self.update_box(self.box)

        # Check if neighbor list needs to be updated
        if self._check_for_nl_update:
            with torch.no_grad():
                self.neighbor_list.update(self.coords, self.box)
                self._check_for_nl_update = False

        # Get pairs from neighbor list (no gradients needed)
        with torch.no_grad():
            pairs = self.neighbor_list.get_pairs()

        # These operations are part of the computational graph and will track gradients
        distance_vecs = self.coords[pairs[:, 1]] - self.coords[pairs[:, 0]]
        distance_vecs = applyPBC(distance_vecs, self.box, self.box_inv)
        dists = torch.linalg.vector_norm(distance_vecs, dim=1)

        return pairs, dists, distance_vecs
    
