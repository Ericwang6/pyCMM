import torch
from typing import Dict

class Parameterizer:
    """
    This class defines an interface for storing and retrieving parameters
    for a force field as well as building the parameter arrays needed
    to evaluate the force field...
    """

    def __init__(self, raw_parameters: Dict, atom_types: torch.Tensor, bond_indices: torch.Tensor) -> None:
        self._parameters = {} # Maps a string to a torch.Tensor
        self._get_axis_frame_indices(raw_parameters, atom_types, bond_indices)
    
    def _get_axis_frame_indices(self, raw_parameters: Dict, atom_types: torch.Tensor, bond_indices: torch.Tensor):
        self._parameters["axistypes"] = raw_parameters["axistypes"][atom_types]
        # TODO: This is only applicable to water. I am not sure exactly how to handle this in general...
        # I don't see how to avoid scalar indexing. Maybe we just need to introduce a concept of axis
        # indices. We could just have the axis types be determined by the topology itself.
        # That would help a lot with setting up the axis systems. Is there any reason not to do this?
        zatoms, xatoms, yatoms = [], [], []
        for i in torch.arange(atom_types.size(0)):
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

        self._parameters["zatoms"] = torch.tensor(zatoms, dtype=torch.long),
        self._parameters["xatoms"] = torch.tensor(xatoms, dtype=torch.long),
        self._parameters["yatoms"] = torch.tensor(yatoms, dtype=torch.long),

    def register_parameters(self, name: str, params: torch.Tensor):
        self._parameters[name] = params

    def checkout_parameters(self, name: str):
        return self._parameters[name]
