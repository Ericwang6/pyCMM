import torch
import math
from iodata import load_one, load_many
from .neighbor_list import *
from typing import Dict, Tuple, List, Optional
from .pbc import applyPBC
from .axis_types import AxisTypes
from .topology import Topology
from .settings import *
from .parameters import Parameterizer2
from .storage import Storage
from .units import BOHR2ANG
from .data import *

#def guess_bonds_from_coordinates_and_labels():


def create_system_from_ext_xyz_file(
        file_name: str, settings: Settings,
        requires_grad: bool=True, requires_box_grad: bool=True,
        device: str="cpu"
    ):
    mols = load_many(file_name, fmt="extxyz")
    systems = []
    for mol in mols:
        atomic_numbers_to_name = {8: "O_water", 1: "H_water"}
        atom_type_names = [atomic_numbers_to_name[mol.atnums[i]] for i in range(len(mol.atnums))]
        box = torch.from_numpy(mol.cellvecs).requires_grad_(requires_box_grad).to(device)
        coords = torch.from_numpy(mol.atcoords).requires_grad_(requires_grad).to(device)
        labels = convert_atomic_numbers_to_labels(mol.atnums)
        bonds = guess_bond_connectivity(mol.atcoords * BOHR2ANG, labels)

        topology = Topology(bonds, coords.size(0), device)
        system = System(coords, box, atom_type_names, topology, settings)
        systems.append(system)
    
    return systems

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
        self.storage = Storage()
        self._check_for_nl_update = False
        self.build_neighbor_list()
        self._find_optimal_ewald_parameters()

    def _find_optimal_ewald_parameters(self):
        lr_elec_settings = self.settings.get_long_range_electrostatics_settings()
        lr_elec_settings.alpha = math.sqrt(-math.log10(2 * lr_elec_settings.tolerance)) / lr_elec_settings.cutoff
        lr_elec_settings.k_max = 50
        for i in range(2, 50):
            error_estimate = (i * math.sqrt(self.box_lengths[0] * lr_elec_settings.alpha) / 20.0) * math.exp(-torch.pi * torch.pi * i * i / (self.box_lengths[0] * lr_elec_settings.alpha * self.box_lengths[0] * lr_elec_settings.alpha))
            if error_estimate < lr_elec_settings.tolerance:
                lr_elec_settings.k_max = i
                break

    def build_neighbor_list(self):
        nl_settings = self.settings.get_neighbor_list_settings()
        if nl_settings.method == "verlet":
            self.neighbor_list = VerletList2(
                self.coords, self.box, nl_settings.cutoff, excluded_atomic_pairs=self.topology.all_intramolecular_pairs,
                padding=torch.tensor(nl_settings.padding / BOHR2ANG, device=self.coords.device)
            )
        # elif nl_settings.method == "something_else":
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
    
    def compute_rotation_matrices(self, z_atoms: torch.Tensor, x_atoms: torch.Tensor, y_atoms: torch.Tensor, axis_types: torch.Tensor):
        """
        Compute local to global rotation matrix for a set of atoms

        Parameters
        ----------
        z_atoms: torch.Tensor[int]
            Atomic indices specifying Z-axis, shape (N,)
        x_atoms: torch.Tensor[int]
            Atomic indices specifying X-axis, shape (N,)
        y_atoms: torch.Tensor[int]
            Atomic indices specifying Y-axis, shape (N,)
        axis_types: torch.Tensor[int]
            Integers specifying local axis types, shape (N,)
        """

        zVec = torch.zeros_like(self.coords)
        xVec = torch.zeros_like(zVec)
        yVec = torch.zeros_like(zVec)

        # Z-Only
        filterZOnly = (axis_types == AxisTypes.ZOnly.value)
        if torch.any(filterZOnly):
            zVec = applyPBC(self.coords[z_atoms][filterZOnly] - self.coords[filterZOnly], self.box, self.box_inv)
            zVecZOnly = torch.nn.functional.normalize(zVec)
            xVecNotZOnly = applyPBC(self.coords[x_atoms][~filterZOnly] - self.coords[~filterZOnly], self.box, self.box_inv)
            xVecZOnly = xVec[~filterZOnly] + torch.nn.functional.normalize(xVecNotZOnly)
            xVecZOnly[filterZOnly, 0] = xVecZOnly[filterZOnly, 0] + 1 - zVecZOnly[filterZOnly, 0]
            xVecZOnly[filterZOnly, 1] = xVecZOnly[filterZOnly, 1] + zVecZOnly[filterZOnly, 0]
            yVecZOnly = torch.nn.functional.normalize(torch.cross(zVecZOnly, xVecZOnly, dim=1))
            zVec[filterZOnly] = zVecZOnly
            xVec[filterZOnly] = xVecZOnly
            yVec[filterZOnly] = yVecZOnly

        # Z-Then-X
        filterZThenX = (axis_types == AxisTypes.ZThenX.value)
        if torch.any(filterZThenX):
            zVecZThenX = torch.nn.functional.normalize(applyPBC(self.coords[z_atoms[filterZThenX]] - self.coords[filterZThenX], self.box, self.box_inv))
            xVecZThenX = applyPBC(self.coords[x_atoms[filterZThenX]] - self.coords[filterZThenX], self.box, self.box_inv)
            xVecZThenX = torch.nn.functional.normalize(xVecZThenX - zVecZThenX * torch.sum(zVecZThenX * xVecZThenX, dim=1, keepdim=True))
            yVecZThenX = torch.nn.functional.normalize(torch.cross(zVecZThenX, xVecZThenX, dim=1))
            zVec[filterZThenX] = zVecZThenX
            xVec[filterZThenX] = xVecZThenX
            yVec[filterZThenX] = yVecZThenX

        # Bisector
        filterBisector = (axis_types == AxisTypes.Bisector.value)
        if torch.any(filterBisector):
            zVecBisector = applyPBC(self.coords[z_atoms[filterBisector]] - self.coords[filterBisector], self.box, self.box_inv)
            xVecBisector = applyPBC(self.coords[x_atoms[filterBisector]] - self.coords[filterBisector], self.box, self.box_inv)
            zVecBisector = torch.nn.functional.normalize(torch.linalg.norm(xVecBisector, dim=1).unsqueeze(1) * zVecBisector + torch.linalg.norm(zVecBisector, dim=1).unsqueeze(1) * xVecBisector)
            xVecBisector = torch.nn.functional.normalize(xVecBisector - zVecBisector * torch.sum(zVecBisector * xVecBisector, dim=1, keepdim=True))
            yVecBisector = torch.nn.functional.normalize(torch.cross(zVecBisector, xVecBisector, dim=1))
            zVec[filterBisector] = zVecBisector
            xVec[filterBisector] = xVecBisector
            yVec[filterBisector] = yVecBisector

        # Z-Bisect
        filterZBisect = (axis_types == AxisTypes.ZBisect.value)
        if torch.any(filterZBisect):
            yVecZBisect = applyPBC(self.coords[y_atoms[filterZBisect]] - self.coords[filterZBisect], self.box, self.box_inv)
            yVecZBisect = torch.nn.functional.normalize(yVecZBisect)
            xVecZBisect = torch.nn.functional.normalize(xVec[filterZBisect] + yVecZBisect)
            zVecZBisect = torch.nn.functional.normalize(torch.cross(xVecZBisect, yVecZBisect, dim=1))
            xVec[filterZBisect] = xVecZBisect
            yVec[filterZBisect] = yVecZBisect
            zVec[filterZBisect] = zVecZBisect

        # Threefold
        # TODO: The way I rewrote this broke this case. Need to fix it.
        #filterThreeFold = (axis_types == AxisTypes.ThreeFold.value)
        #if torch.any(filterThreeFold):
        #    yVecThreeFold = applyPBC(self.coords[y_atoms][filterThreeFold] - self.coords[filterThreeFold], self.box, self.box_inv)
        #    yVecThreeFold = torch.nn.functional.normalize(yVecThreeFold)
        #    xVecThreeFold = xVec[filterThreeFold]
        #    zVecThreeFold = zVec[filterThreeFold]
        #    zVec[filterThreeFold] = torch.nn.functional.normalize(zVecThreeFold + xVecThreeFold + yVecThreeFold)

        # No axis
        filterNoAxis = (axis_types == AxisTypes.NoAxisType.value)
        if torch.any(filterNoAxis):
            xVecNoAxis = torch.tensor([1.0, 0.0, 0.0], device=zVec.device)
            yVecNoAxis = torch.tensor([0.0, 1.0, 0.0], device=zVec.device)
            zVecNoAxis = torch.tensor([0.0, 0.0, 1.0], device=zVec.device)
            xVec[filterNoAxis] = xVec[filterNoAxis] + xVecNoAxis
            yVec[filterNoAxis] = yVec[filterNoAxis] + yVecNoAxis
            zVec[filterNoAxis] = zVec[filterNoAxis] + zVecNoAxis

        rotMatrix = torch.hstack((xVec, yVec, zVec)).reshape(-1, 3, 3)
        return rotMatrix