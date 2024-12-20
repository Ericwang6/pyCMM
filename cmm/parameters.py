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

    def __init__(self, atom_type_names: List[str], pairs: torch.Tensor, angle_pairs: torch.Tensor, raw_atomic_params: Dict[str, torch.Tensor], raw_bonded_pair_params: Dict[Tuple[str, str], torch.Tensor]) -> None:
        self._atomic_param_arrays = {} # Map from parameter type to array indexed by atom type
        self._pair_param_arrays = {} # Map from parameter type to array indexed by pair type

        # Used for determining atom types. Initial build is on CPU currently.
        self._atom_type_names = atom_type_names
        self._unique_atom_type_names = list(set(atom_type_names))

        self._define_atom_and_pair_type_names_and_indices()
        self._flatten_raw_parameter_dicts_to_arrays(raw_atomic_params, raw_bonded_pair_params)
        self._build_atomic_parameter_arrays(raw_atomic_params)
        self._build_pair_parameter_arrays(raw_bonded_pair_params)

        self._atom_types = torch.tensor([self._name_to_atom_type[name] for name in self._atom_type_names], dtype=torch.long) # On device
        self._pair_types = self._symmetric_pairing_function(self._atom_types[pairs])
        
        angle_type_pairs = torch.vmap(lambda x: self._symmetric_pairing_function(self._atom_types[pairs[x]]))(angle_pairs)
        self._angle_types = self._symmetric_pairing_function(angle_type_pairs.T)

    def _define_atom_and_pair_type_names_and_indices(self):
        self._name_to_atom_type = {}
        for i in range(len(self._unique_atom_type_names)):
            self._name_to_atom_type[self._unique_atom_type_names[i]] = i

        all_type_combos = torch.combinations(torch.arange(len(self._unique_atom_type_names)), with_replacement=True)
        self._unique_pair_types = self._symmetric_pairing_function(all_type_combos)
        self._unique_pair_type_names = [(self._unique_atom_type_names[all_type_combos[i][0]], self._unique_atom_type_names[all_type_combos[i][1]]) for i in torch.arange(len(all_type_combos))]
        self._name_to_pair_type = {}
        for i in range(len(self._unique_pair_type_names)):
            self._name_to_pair_type[self._unique_pair_type_names[i]] = self._unique_pair_types[i]

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

        NOTE(JOE): It is possible that at some point we will actually want a non-symmetric
        pairing function since conceivably we could have parameters that differ depending
        on the order of the parameter. I think if that situation ever arises, we are
        probably just making a bad decision. Physically, I am not sure how this situation
        would arise. If it does, though, this problem can be circumvented by just making
        an additional parameter. This is basically what we do with the CT_acceptor and CT_donor
        parameters.
        """
        k = torch.floor_divide(torch.square(torch.sum(pairs, dim=1) + 1) - torch.remainder((torch.sum(pairs, dim=1) + 1), 2), 4) + torch.min(pairs, dim=1).values
        return k

    def _flatten_raw_parameter_dicts_to_arrays(self, raw_atomic_params: Dict[str, torch.Tensor], raw_bonded_pair_params: Dict[Tuple[str, str], torch.Tensor]):
        # NOTE(JOE): Currently there is some stuff happening here with strings
        # so I am using regular python. Need to figure out how to use strings
        # with pytorch if it is even possible.
        # LATER: Will want to use torchtext.vocab to create integer encodings.
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

        for i in range(len(self._unique_pair_type_names)):
            if self._unique_pair_type_names[i] in raw_bonded_pair_params:
                for param_key in raw_bonded_pair_params[self._unique_pair_type_names[i]]:
                    # Just have to pick a size larger than the maximum pair type.
                    # Eventually we may want to convert this to just be a sparse tensor.
                    # The actual number for angle types and dihedral types could get
                    # fairly large, so we should switch to sparse storage format then.
                    self._pair_param_arrays[param_key] = torch.zeros(n_types * n_types)
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
