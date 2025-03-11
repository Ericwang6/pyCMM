from .neighbor_list import *
import torch
from typing import Dict, Tuple
from .topology import Topology

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
        self.box_lengths = torch.diag(box)
        self.neighbor_list = CellList(coords, self.box_lengths, cutoff, max_neighbors=max_neighbors)

    def get_distances_vectors_and_pairs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all distances, distance vectors, and indices of atom pairs.
        """
        # @SPEED: It's unclear if I should deal with pairs in this manner or as its
        # transpose. I suppose it probably doens't matter since I end up using both
        # but if I could avoid that, then that would be ideal. I guess I could
        # transpose the coordinate to solve this problem?
        self.pairs = self.neighbor_list.get_pairs()
        distance_vecs = self.coords[self.pairs[:, 1]] - self.coords[self.pairs[:, 0]]
        self.distance_vecs = distance_vecs - torch.round(distance_vecs / self.box_lengths) * self.box_lengths
        self.dists = torch.linalg.vector_norm(distance_vecs, dim=1)
        return self.pairs, self.dists, self.distance_vecs
