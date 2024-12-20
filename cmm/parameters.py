import torch
from typing import Dict, List, Tuple

class Parameterizer:
    """
    This class defines an interface for storing and retrieving parameters
    for a force field as well as building the parameter arrays needed
    to evaluate the force field. The parameterizer will look at the actual
    atom type names being used in the calculation and convert these into
    atom types which are used to index the parameter arrays provided by
    the force field. In this way, the parameterizer builds the appropriate
    size tensors for the potential evaluation and fills them with the right
    parameters.
    """

    def __init__(self, atom_type_names: List[str], pairs: torch.Tensor, angle_pairs: torch.Tensor, raw_atomic_params: Dict[str, torch.Tensor], raw_bonded_pair_params: Dict[Tuple[str, str], torch.Tensor], raw_angle_params: Dict[Tuple[str, str, str], torch.Tensor]) -> None:
        self._atomic_param_arrays = {} # Map from parameter type to array indexed by atom type
        self._pair_param_arrays = {} # Map from parameter type to array indexed by pair type
        self._angle_param_arrays = {} # Map from parameter type to array indexed by angle type

        # Used for determining atom types. Initial build is on CPU currently.
        self._atom_type_names = atom_type_names
        self._unique_atom_type_names = list(dict.fromkeys(atom_type_names)) # trick to create ordered set

        self._define_type_names_and_indices()
        self._flatten_raw_parameter_dicts_to_arrays(raw_atomic_params, raw_bonded_pair_params, raw_angle_params)
        self._build_atomic_parameter_arrays(raw_atomic_params)
        self._build_pair_parameter_arrays(raw_bonded_pair_params)
        self._build_angle_parameter_arrays(raw_angle_params)

        self._atom_types = torch.tensor([self._name_to_atom_type[name] for name in self._atom_type_names], dtype=torch.long) # On device
        self._pair_types = self._symmetric_pairing_function(self._atom_types[pairs])
        self._angle_types = self._nonsymmetric_pairing_function(torch.column_stack((self._nonsymmetric_pairing_function(self._atom_types[pairs[angle_pairs[0]]]), self._nonsymmetric_pairing_function(self._atom_types[pairs[angle_pairs[1]]]))))

    def _define_type_names_and_indices(self):
        # Atom types #
        self._name_to_atom_type = {}
        for i in range(len(self._unique_atom_type_names)):
            self._name_to_atom_type[self._unique_atom_type_names[i]] = i

        # Pair types #
        self._name_to_pair_type = {}
        all_type_combos = torch.combinations(torch.arange(len(self._unique_atom_type_names)), 2, with_replacement=True)
        self._unique_pair_types = self._symmetric_pairing_function(all_type_combos)
        self._unique_pair_type_names = [(self._unique_atom_type_names[all_type_combos[i][0]], self._unique_atom_type_names[all_type_combos[i][1]]) for i in torch.arange(len(all_type_combos))]
        for i in range(len(self._unique_pair_type_names)):
            self._name_to_pair_type[self._unique_pair_type_names[i]] = self._unique_pair_types[i]
        
        # Angle types #
        self._name_to_angle_type = {}
        self._unique_angle_type_names = [[(self._unique_atom_type_names[all_type_combos[i][0]], self._unique_atom_type_names[j], self._unique_atom_type_names[all_type_combos[i][1]]) for i in torch.arange(len(all_type_combos))] for j in torch.arange(len(self._unique_atom_type_names))]
        self._unique_angle_type_names = [item for sublist in self._unique_angle_type_names for item in sublist]
        self._atom_types_in_angle = torch.tensor([(self._name_to_atom_type[self._unique_angle_type_names[i][0]], self._name_to_atom_type[self._unique_angle_type_names[i][1]], self._name_to_atom_type[self._unique_angle_type_names[i][2]]) for i in torch.arange(len(self._unique_angle_type_names))])
        angle_pair_1 = self._atom_types_in_angle[:, [1, 0]]
        angle_pair_2 = self._atom_types_in_angle[:, [1, 2]]
        angle_pair_types_1 = self._nonsymmetric_pairing_function(angle_pair_1)
        angle_pair_types_2 = self._nonsymmetric_pairing_function(angle_pair_2)
        self._unique_angle_types = self._nonsymmetric_pairing_function(torch.column_stack((angle_pair_types_1, angle_pair_types_2)))
        for i in range(len(self._unique_angle_type_names)):
            self._name_to_angle_type[self._unique_angle_type_names[i]] = self._unique_angle_types[i]

    def _symmetric_pairing_function(self, pairs: torch.Tensor) -> torch.Tensor:
        """
        Given two positive indices, (i,j), this function generates a unique index k.
        The particular pairing function chosen here is described in: https://arxiv.org/pdf/2105.10752
        The important features of this pairing function are that it is symmetric (i.e. the order of
        indices does not matter. Most pairing functions intentionally don't have this property).
        Additionally, if we have N atom types, the largest index is close to N^2. We can check
        the actual value and use it to pre-allocate the arrays we index into. In the case that
        the number of atom types is very large, the arrays will become sparse and we can just
        revisit this solution at that point. Likely, just using a sparse arrays is sufficient.
        """
        k = torch.floor_divide(torch.square(torch.sum(pairs, dim=1) + 1) - torch.remainder((torch.sum(pairs, dim=1) + 1), 2), 4) + torch.min(pairs, dim=1).values
        return k
    
    def _nonsymmetric_pairing_function(self, pairs: torch.Tensor) -> torch.Tensor:
        """
        Given two positive indices, (i,j), this function generates a unique index k.
        The index is different for (j,i) and (i,j). Used for generating the angle and dihedral types.
        This is the "ElegantPair" function defined at: https://en.wikipedia.org/wiki/Pairing_function#Other_pairing_functions
        """
        pair_types = torch.zeros(pairs.size(0), dtype=torch.long)
        i_less_than_j_indices = torch.where(pairs[:, 0] < pairs[:, 1])
        other_indices = torch.where(pairs[:, 0] >= pairs[:, 1])
        pair_types[i_less_than_j_indices] = torch.square(pairs[i_less_than_j_indices][:, 1]) + pairs[i_less_than_j_indices][:, 0]
        pair_types[other_indices] = torch.square(pairs[other_indices][:, 0]) + pairs[other_indices][:, 0] + pairs[other_indices][:, 1]
        return pair_types

    def _flatten_raw_parameter_dicts_to_arrays(self, raw_atomic_params: Dict[str, torch.Tensor], raw_bonded_pair_params: Dict[Tuple[str, str], torch.Tensor], raw_angle_params: Dict[Tuple[str, str, str], torch.Tensor]):
        # Atom Types #
        n_types = len(self._unique_atom_type_names)
        for param_key in raw_atomic_params[self._unique_atom_type_names[0]]:
            self._atomic_param_arrays[param_key] = torch.zeros_like(raw_atomic_params[self._unique_atom_type_names[0]][param_key])
            self._atomic_param_arrays[param_key].unsqueeze_(0)
            if self._atomic_param_arrays[param_key].ndim == 1: # Floats: e.g. charges
                self._atomic_param_arrays[param_key] = self._atomic_param_arrays[param_key].repeat(n_types, 1)
            elif self._atomic_param_arrays[param_key].ndim == 2: # Vectors: e.g. dipole moment
                self._atomic_param_arrays[param_key] = self._atomic_param_arrays[param_key].repeat(n_types, 1, 1)
            elif self._atomic_param_arrays[param_key].ndim == 3: # Matrices: e.g. polarizability
                self._atomic_param_arrays[param_key] = self._atomic_param_arrays[param_key].repeat(n_types, 1, 1, 1)

        # @SPEED: The pair, angle, and dihedral types are currently expressed so that we index
        # into a dense array. Many indices do not correspond to a valid pair, angle, or dihedral type
        # for a given system. i.e. in water, we define all angle types for the elements O and H, but
        # only HOH angles are actually needed.
        # There are two options for fixing this, which may be faster/use-less-memory when the number of atom types
        # is large. One is to make the types dense by using a map that takes the computed types and
        # turns puts them in the range 0:N_pair_types, etc. The other option to turn these dense arrays
        # into sparse arrays whose only valid indices are the pair types, angle types, etc.

        # Pair Types #
        for i in range(len(self._unique_pair_type_names)):
            if self._unique_pair_type_names[i] in raw_bonded_pair_params:
                for param_key in raw_bonded_pair_params[self._unique_pair_type_names[i]]:
                    self._pair_param_arrays[param_key] = torch.zeros(torch.max(self._unique_pair_types))
                break
        
        # Angle Types #
        for i in range(len(self._unique_angle_type_names)):
            if self._unique_angle_type_names[i] in raw_angle_params:
                for param_key in raw_angle_params[self._unique_angle_type_names[i]]:
                    self._angle_param_arrays[param_key] = torch.zeros(torch.max(self._unique_angle_types))
                break

    def _build_atomic_parameter_arrays(self, raw_atomic_params: Dict[str, torch.Tensor]):
        for param_key in self._atomic_param_arrays.keys():
            for i in torch.arange(len(self._unique_atom_type_names)):
                self._atomic_param_arrays[param_key][i] = raw_atomic_params[self._unique_atom_type_names[i]][param_key]

    def _build_pair_parameter_arrays(self, raw_bonded_pair_params: Dict[Tuple[str, str], torch.Tensor]):
        for param_key in self._pair_param_arrays.keys():
            for i in torch.arange(len(self._unique_pair_type_names)):
                if self._unique_pair_type_names[i] in raw_bonded_pair_params:
                    self._pair_param_arrays[param_key][self._name_to_pair_type[self._unique_pair_type_names[i]]] = raw_bonded_pair_params[self._unique_pair_type_names[i]][param_key][0]

    def _build_angle_parameter_arrays(self, raw_angle_params: Dict[Tuple[str, str, str], torch.Tensor]):
        for param_key in self._angle_param_arrays.keys():
            for i in torch.arange(len(self._unique_angle_type_names)):
                if self._unique_angle_type_names[i] in raw_angle_params:
                    self._angle_param_arrays[param_key][self._name_to_angle_type[self._unique_angle_type_names[i]]] = raw_angle_params[self._unique_angle_type_names[i]][param_key][0]

    def _get_axis_frame_indices(self, atom_types: torch.Tensor):
        #self._parameters["axistypes"] = raw_parameters["axistypes"][atom_types]
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

    def get_atomic_parameters(self, name: str):
        return self._atomic_param_arrays[name][self._atom_types].squeeze_()
    
    def get_pair_parameters(self, name: str, pairs: torch.Tensor):
        return self._pair_param_arrays[name][self._pair_types[pairs]]
    
    # NOTE(JOE): The method below does not require the pairs since all the angles
    # that are in the system which need to be evaluated are inferred and their types
    # stored when this object is constructed. If we ever needed to accomodate
    # non-bonded angles, then we need a second method where we compute the angle types
    # specified by pairs of pairs. Then return the appropriate parameters.
    # No such things exist in CMM, so we don't implement that, but there are h-bond
    # potential functions out there which are angular in nature but non-bonded.
    def get_angle_parameters(self, name: str):
        return self._angle_param_arrays[name][self._angle_types]
