import torch
from typing import Dict

class Parameterizer:
    """
    This class defines an interface for storing and retrieving parameters
    for a force field as well as building the parameter arrays needed
    to evaluate the force field...

    TODO: In the future, this SHOULD NOT require the positions and box in order
    to be constructed. Specifically, we end up re-computing a small number of
    distances here which can be avoided. In order to do so, the indices for
    the evaluation of the axis frames needs to be in terms of the distance
    vectors not the atomic positions. This is likely a small optimization.
    """

    def __init__(self, raw_parameters: Dict, atom_types: torch.Tensor) -> None:
        self._parameters = {} # Maps a string to a torch.Tensor
        self._get_axis_frame_indices(raw_parameters, atom_types)
        self._fill_parameter_dictionary_water(raw_parameters, atom_types, int(atom_types.size(0) / 3))
        
    
    def _get_axis_frame_indices(self, raw_parameters: Dict, atom_types: torch.Tensor):
        self._parameters["axistypes"] = raw_parameters["axistypes"][atom_types]
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

        self._parameters["zatoms"] = torch.tensor(zatoms, dtype=torch.long)
        self._parameters["xatoms"] = torch.tensor(xatoms, dtype=torch.long)
        self._parameters["yatoms"] = torch.tensor(yatoms, dtype=torch.long)

    def _fill_parameter_dictionary_water(self, raw_parameters: Dict, atom_types: torch.Tensor, num_waters: int):
        # This is a strictly temporary method while the more generic approach using
        # bond types and so on is implemented.
        for key in raw_parameters:
            if key == 'eps':
                self._parameters[key] = raw_parameters[key][torch.meshgrid(atom_types, atom_types, indexing='xy')]
            elif key in ['k_b', 'r_eq', 'beta', 'k_ba', 'D',
                       'j_cf_pauli', 'j_cf',  'k_hardness_b',
                       'dip_deriv_1', 'dip_deriv_2', 'ct_slope_1', 'ct_slope_2']:
                self._parameters[key] = raw_parameters[key][torch.zeros(num_waters * 2, dtype=torch.long)]
            elif key in ['k_bb', 'theta_eq', 'k_theta', 'j_cf_bb', 'j_cf_angle', 'k_hardness_bb', 'k_hardness_angle']:
                self._parameters[key] = raw_parameters[key][torch.zeros(num_waters, dtype=torch.long)]
            else:
                self._parameters[key] = raw_parameters[key][atom_types]

    def _fill_parameter_dictionary(self, raw_parameters: Dict, atom_types: torch.Tensor, bond_indices: torch.Tensor):
        # TODO: The above implementation is only applicable to water right now.
        # We need to come up with a more general way of dealing with
        # coupling parameters specifically. I think we need to
        # introduce a "bond type" concept which specifies which
        # bond we are looking in terms of a uniquely defined index.
        # Similarly, we can have an angle type, dihedral type.
        # We can then find the bond-bond and bond-angle
        # and bond-dihedral parameters from the combinations of these
        # types. Exactly how this will work requires some thought.
        
        # The parameter arrays should constructed by indexing over the atom types
        # bond types, and so on. 
        pass

    def register_parameters(self, name: str, params: torch.Tensor):
        self._parameters[name] = params

    def checkout_parameters(self, name: str):
        return self._parameters[name]
