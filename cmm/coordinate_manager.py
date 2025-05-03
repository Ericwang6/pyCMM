from .neighbor_list import *
import torch
from typing import Dict, Tuple, List, Optional
from .pbc import applyPBC
from .axis_types import AxisTypes
from .topology import Topology

class CoordinateManager:
    def __init__(self, coords: torch.Tensor, box: torch.Tensor, cutoff: float, labels: List[str], max_neighbors: int = 1024) -> None:
        self._need_coordinate_grads = coords.requires_grad
        self._need_box_grads = box.requires_grad
        self.coords = coords
        self.box = box
        self.labels = labels
        self.box_inv = torch.inverse(self.box)
        self.box_lengths = torch.diagonal(self.box)
        self.box_volume = torch.det(self.box)
        self.cutoff = torch.tensor(cutoff)
        self.max_neighbors = max_neighbors
        if cutoff > 0.5 * min(self.box_lengths):
            print(f"Requested cutoff of {cutoff:.4f} is larger than half of the smallest side length {0.5 * min(self.box_lengths):.4f}. Setting the cutoff to {0.5 * min(self.box_lengths):.4f}")
            self.cutoff = 0.5 * min(self.box_lengths).detach()
        with torch.no_grad():
            self.neighbor_list = VerletList(coords, self.box, self.cutoff, padding=1.0)
        self._check_for_nl_update = False

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
        if self.cutoff > 0.5 * min(self.box_lengths):
            print(f"WARNING: Requested cutoff of {self.cutoff:.4f} is larger than half of the smallest side length {0.5 * min(self.box_lengths):.4f}.")
            #self.cutoff = 0.5 * min(self.box_lengths).detach()
            #self.neighbor_list.cutoff = self.cutoff
            #self.neighbor_list.verlet_cutoff = self.cutoff + self.neighbor_list.cutoff_padding
        
        self._check_for_nl_update = True
        
    def get_distances_vectors_and_pairs(self, topology: Topology = None, reset_grads=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                topology.rebuild(self.neighbor_list)
                self._check_for_nl_update = False

        # Get pairs from neighbor list (no gradients needed)
        with torch.no_grad():
            self.pairs = self.neighbor_list.get_pairs()

        # These operations are part of the computational graph and will track gradients
        distance_vecs = self.coords[self.pairs[:, 1]] - self.coords[self.pairs[:, 0]]
        self.distance_vecs = applyPBC(distance_vecs, self.box, self.box_inv)
        self.dists = torch.linalg.vector_norm(self.distance_vecs, dim=1)

        return self.pairs, self.dists, self.distance_vecs

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
