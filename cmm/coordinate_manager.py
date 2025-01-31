from .neighbor_list import *
import torch
from typing import Dict, Tuple
from .topology import Topology
from .pbc import applyPBC
from .axis_types import AxisTypes

# One purpose of having a CoordinateMananager as a concept is
# to make it easy to ensure that if we sort coordinates to
# enhance locality, that we do not need to worry about the
# NeighborList getting out of sync since this keeps everything to
# do with coordinates in sync.
# 
# Additionally, the coordinate manager hands off the distance vectors,
# pairs, and so on in such a way that pair potentials don't have to
# do anything but apply the potential to all the distances.
# The potentials will not even see the neighbor list. They are just
# functions that take in bond lengths, vectors, angles, or whatever, 
# along with a set of parameters. The coordinate manager enables that
# separation.
#
# Finally, this ends up being a convenient place to deal with different
# box types, whether or not to apply PBCs, and so on.

class CoordinateManager:
    def __init__(self, coords: torch.Tensor, box: torch.Tensor, cutoff: float, max_neighbors: int = 512) -> None:
        self._need_coordindate_grads = coords.requires_grad
        self._need_box_grads = box.requires_grad
        self.coords = coords
        self.box = box
        self.cutoff = torch.tensor(cutoff)
        self.box_inv = torch.inverse(self.box)
        self.box_lengths = torch.diagonal(self.box)
        self.neighbor_list = CellList(coords, self.box_lengths, self.cutoff, max_neighbors=max_neighbors)
        self._check_for_nl_update = False

    def update_coordinates(self, new_coords: torch.Tensor):
        """
        Update coordinates held by the coordinate manager and neighbor list.
        """
        self.coords = new_coords.detach().clone().requires_grad_()
        self._check_for_nl_update = True

    def update_box(self, new_box: torch.Tensor):
        """
        Update coordinates held by the coordinate manager and neighbor list.
        """
        self.box = new_box.detach().clone().requires_grad_()
        self.box_inv = torch.inverse(self.box)
        self.box_lengths = torch.diagonal(self.box)
        self.neighbor_list.box_lengths = self.box_lengths
        self._check_for_nl_update = True

    def get_distances_vectors_and_pairs(self, reset_gradients: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all distances, distance vectors, and indices of atom pairs.
        """
        # @SPEED: It's unclear if I should deal with pairs in this manner or as its
        # transpose. I suppose it probably doens't matter since I end up using both
        # but if I could avoid that, then that would be ideal. I guess I could
        # transpose the coordinate to solve this problem?
        
        if reset_gradients:
            self.update_coordinates(self.coords)
            self.update_box(self.box)
        # TODO: Something is broken about NL rebuilds!
        # Fix that!
        #if self._check_for_nl_update:
        #    self.neighbor_list.update(self.coords, self.box_lengths)
        self.pairs = self.neighbor_list.get_pairs()
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

        zVec = applyPBC(self.coords[z_atoms] - self.coords, self.box, self.box_inv)
        zVec = torch.nn.functional.normalize(zVec)
        xVec = torch.zeros_like(zVec)
        yVec = torch.zeros_like(zVec)

        # Z-Only
        filterZOnly = (axis_types == AxisTypes.ZOnly.value)
        xVecNotZOnly = applyPBC(self.coords[x_atoms][~filterZOnly] - self.coords[~filterZOnly], self.box, self.box_inv)
        xVec[~filterZOnly] += torch.nn.functional.normalize(xVecNotZOnly)
        xVec[filterZOnly, 0] += 1 - zVec[filterZOnly, 0]
        xVec[filterZOnly, 1] += zVec[filterZOnly, 0]

        # Bisector
        filterBisector = (axis_types == AxisTypes.Bisector.value)
        if torch.any(filterBisector):
            zVec[filterBisector] += xVec[filterBisector]
            zVec = torch.nn.functional.normalize(zVec)

        # Z-Bisect
        filterZBisect = (axis_types == AxisTypes.ZBisect.value)
        if torch.any(filterZBisect):
            yVecZBisect = applyPBC(self.coords[y_atoms][filterZBisect] - self.coords[filterZBisect], self.box, self.box_inv)
            yVecZBisect = torch.nn.functional.normalize(yVecZBisect)
            xVecZBisect = torch.nn.functional.normalize(xVec[filterZBisect] + yVecZBisect)
            xVec[filterZBisect] = xVecZBisect

        # Threefold
        filterThreeFold = (axis_types == AxisTypes.ThreeFold.value)
        if torch.any(filterThreeFold):
            yVecThreeFold = applyPBC(self.coords[y_atoms][filterThreeFold] - self.coords[filterThreeFold], self.box, self.box_inv)
            yVecThreeFold = torch.nn.functional.normalize(yVecThreeFold)
            xVecThreeFold = xVec[filterThreeFold]
            zVecThreeFold = zVec[filterThreeFold]
            zVec[filterThreeFold] = torch.nn.functional.normalize(zVecThreeFold + xVecThreeFold + yVecThreeFold)

        xVec = torch.nn.functional.normalize(xVec - zVec * torch.sum(zVec * xVec, dim=1, keepdim=True))
        yVec = torch.linalg.cross(zVec, xVec)

        # No axis
        filterNoAxis = (axis_types == AxisTypes.NoAxisType.value)
        if torch.any(filterNoAxis):
            zVec[filterNoAxis] = torch.tensor([0.0, 0.0, 1.0])
            xVec[filterNoAxis] = torch.tensor([1.0, 0.0, 0.0])
            yVec[filterNoAxis] = torch.tensor([0.0, 1.0, 0.0])

        rotMatrix = torch.hstack((xVec, yVec, zVec)).reshape(-1, 3, 3)
        return rotMatrix
