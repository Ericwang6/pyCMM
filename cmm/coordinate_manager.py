from .neighbor_list import *
import torch
from typing import Dict, Tuple

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

        #self.distance_vectors = self._compute_distance_vectors()

    def get_intermolecular_distances_vectors_and_pairs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all distances, distance vectors, and indices of atom pairs
        needed to compute intermolecular interactions.
        """
        # TODO: Mask off the intramolecular pairs. Probably all of this should
        # happen in the constructor. We also need an update function of some kind.
        # After splitting the pairs into intra and inter, check you get the same
        # numbers when swapping in these functions for computing the pairs, distances,
        # and so on.
        pairs = self.neighbor_list.get_pairs()
        distance_vecs = self.coords[pairs[1]] - self.coords[pairs[0]]
        distance_vecs = distance_vecs - torch.round(distance_vecs / self.box_lengths) * self.box_lengths
        dists = torch.linalg.vector_norm(distance_vecs, dim=1)
        return pairs, dists, distance_vecs

        #return self.neighbor_list._get_distances_and_vectors()
