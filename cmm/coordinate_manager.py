from .neighbor_list import *
import torch
from typing import Dict, Tuple
from .topology import Topology
from .pbc import applyPBC

# One purpose of having a CoordinateMananager as a concept is
# to make it easy to ensure that if we sort coordinates to
# enhance locality, that we do not need to worry about the
# NeighborList getting out of sync since this keeps everything to
# do with coordinates in sync.
# 
# Additionally, the coordinate manager hands off the distance vectors,
# angles, and so on in such a way that pair potentials don't have to
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
        # TODO: Allow for multiple cutoffs since short_range only needs roughly
        # 6 angstroms but vdw needs 12 angstroms usually. Also allow for choice
        # of neighbor list. We just use a cell list for now.
        self.coords = coords
        self.box = box
        self.box_inv = torch.inverse(box)
        self.box_lengths = torch.diag(box)
        self.neighbor_list = CellList(coords, self.box_lengths, cutoff, max_neighbors=max_neighbors)
        self._get_axis_frame_indices()

    def _get_axis_frame_indices(self):
        # TODO: This is only applicable to water. I am not sure exactly how to handle this in general...
        # I don't see how to avoid scalar indexing. Maybe we just need to introduce a concept of axis
        # indices. We could just have the axis types be determined by the topology itself.
        # That would help a lot with setting up the axis systems. Is there any reason not to do this?
        # Could also store the axis indices as an Nx3 set of indices where entries hold the
        # distance vector index needed to compute the axis system and the axis system is
        # determined by the type. Could do similar with an Nx4 set of indices to the atoms themselves.
        # That lets the atom types be determined by the force field.
        # Could parse the 1-2, 1-3, and 1-4 indices. That is what actually gets passed to the
        # topology to build stuff. Those indices are what's needed to build this.
        zatoms, xatoms, yatoms = [], [], []
        for i in torch.arange(self.coords.size(0)):
            if i % 3 == 0:
                zatoms.append(i + 1)
                xatoms.append(i + 2)
            elif i % 3 == 1:
                zatoms.append(i - 1)
                xatoms.append(i + 1)
            else:
                zatoms.append(i - 2)
                xatoms.append(i - 1)
            yatoms.append(-1)

        # TODO: This should probably go in the coordinate manager.
        self._zatoms = torch.tensor(zatoms, dtype=torch.long)
        self._xatoms = torch.tensor(xatoms, dtype=torch.long)
        self._yatoms = torch.tensor(yatoms, dtype=torch.long)

    def get_distances_vectors_and_pairs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all distances, distance vectors, and indices of atom pairs.
        """
        # @SPEED: It's unclear if I should deal with pairs in this manner or as its
        # transpose. I suppose it probably doens't matter since I end up using both
        # but if I could avoid that, then that would be ideal. I guess I could
        # transpose the coordinate to solve this problem?
        self.pairs = self.neighbor_list.get_pairs()
        self.distance_vecs = self.coords[self.pairs[:, 1]] - self.coords[self.pairs[:, 0]]
        # SOMEHOW I GET THE RIGHT ANSWER WHEN DOING ABOVE WHICH DOESNT RESPECTS PBCS???????
        #self.distance_vecs = distance_vecs - torch.round(distance_vecs / self.box_lengths) * self.box_lengths
        #self.distance_vecs = applyPBC(distance_vecs, self.box, self.box_inv)
        self.dists = torch.linalg.vector_norm(self.distance_vecs, dim=1)
        return self.pairs, self.dists, self.distance_vecs

    def compute_rotation_matrices(self, axis_types: torch.Tensor):
        """
        Compute local to global rotation matrix.
        Axis types are specified as follows:
        0 - Identity
        1 - Z-Then-X
        2 - Bisector

        TODO: Need to explicitly deal with the axis types for ions (Identity). Currently, we
        assume the two possible types are really Z-Then-X and Bisector. Basically, torch select
        on the axis type.
        """
        coords_z_axis = self.coords[self._zatoms]
        coords_x_axis = self.coords[self._xatoms]
        coords_y_axis = self.coords[self._yatoms] # Unused since neither available axis system needs this info.
        # WARN(JOE): The above is a silent bug waiting to happen. The NULL value stored for yatoms is -1
        # which will grab the last index of coordinates. Whenever additional axis types get implemented
        # that actually use the y-coordinate, we just have to make sure that data doesn't accidentally get
        # used for axis types that don't use the y-coordinates. (i.e. do not assume those vectors will be zeros)
        # I think just torch.select on everything not equal to -1.

        # ZThenX
        zvec = coords_z_axis - self.coords
        xvec = coords_x_axis - self.coords
        zvec = torch.nn.functional.normalize(zvec - torch.round(zvec / self.box_lengths) * self.box_lengths)
        xvec = torch.nn.functional.normalize(xvec - torch.round(xvec / self.box_lengths) * self.box_lengths)
        
        # Bisector  
        zvec += xvec * (axis_types == 2).unsqueeze(1)
        zvec = torch.nn.functional.normalize(zvec)

        xvec = xvec - torch.sum(zvec * xvec, dim=1, keepdim=True) * zvec
        xvec = torch.nn.functional.normalize(xvec)
        yvec = torch.linalg.cross(zvec, xvec)
        rotMatrix = torch.hstack((xvec, yvec, zvec)).reshape(-1, 3, 3)
        return rotMatrix
