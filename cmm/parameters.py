import torch
from typing import Dict, List, Tuple, Optional

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

    def __init__(self, atom_type_names: List[str], pairs: torch.Tensor, angle_atoms: torch.Tensor, raw_atomic_params: Dict[str, torch.Tensor], raw_pair_params: Dict[Tuple[str, str], torch.Tensor], raw_pair_pair_params: Dict[Tuple[Tuple[str, str], Tuple[str, str]], torch.Tensor], raw_pair_angle_params: Dict[Tuple[Tuple[str, str], Tuple[str, str, str]], torch.Tensor], raw_angle_params: Dict[Tuple[str, str, str], torch.Tensor]) -> None:
        self.device = pairs.device
        self._atomic_param_arrays = {} # Map from parameter type to array indexed by atom type
        self._pair_param_arrays = {} # Map from parameter type to array indexed by pair type
        self._pair_pair_param_arrays = {} # Map from parameter type to array indexed by two pair types
        self._pair_angle_param_arrays = {} # Map from parameter type to array indexed by a pair and angle type
        self._angle_param_arrays = {} # Map from parameter type to array indexed by angle type

        # Used for determining atom types. Initial build is on CPU currently.
        self._atom_type_names = atom_type_names
        self._unique_atom_type_names = list(dict.fromkeys(atom_type_names)) # trick to create ordered set

        self._define_type_names_and_indices()
        
        self._flatten_raw_parameter_dicts_to_arrays(raw_atomic_params, raw_pair_params, raw_pair_pair_params, raw_pair_angle_params, raw_angle_params)
        self._build_atomic_parameter_arrays(raw_atomic_params)
        self._build_pair_parameter_arrays(raw_pair_params)
        self._build_pair_pair_parameter_arrays(raw_pair_pair_params)
        self._build_pair_angle_parameter_arrays(raw_pair_angle_params)
        self._build_angle_parameter_arrays(raw_angle_params)

        self._atom_types = torch.tensor([self._name_to_atom_type[name] for name in self._atom_type_names], dtype=torch.long, device=self.device)
        self._pair_types = self._symmetric_pairing_function(self._atom_types[pairs])
        self._angle_types = self._get_angle_types_from_angle_atoms(angle_atoms)

    def rebuild(self, pairs: torch.Tensor, raw_atomic_params: Dict[str, torch.Tensor], raw_pair_params: Dict[Tuple[str, str], torch.Tensor], raw_pair_pair_params: Dict[Tuple[Tuple[str, str], Tuple[str, str]], torch.Tensor], raw_pair_angle_params: Dict[Tuple[Tuple[str, str], Tuple[str, str, str]], torch.Tensor], raw_angle_params: Dict[Tuple[str, str, str], torch.Tensor]):
        self._flatten_raw_parameter_dicts_to_arrays(raw_atomic_params, raw_pair_params, raw_pair_pair_params, raw_pair_angle_params, raw_angle_params)
        self._pair_types = self._symmetric_pairing_function(self._atom_types[pairs])
        self._build_atomic_parameter_arrays(raw_atomic_params)
        self._build_pair_parameter_arrays(raw_pair_params)
        self._build_pair_pair_parameter_arrays(raw_pair_pair_params)
        self._build_pair_angle_parameter_arrays(raw_pair_angle_params)
        self._build_angle_parameter_arrays(raw_angle_params)

    def _get_angle_types_from_angle_atoms(self, angle_atoms: torch.Tensor):
        if angle_atoms.numel() > 0:
            return self._nonsymmetric_pairing_function(torch.column_stack((self._nonsymmetric_pairing_function(self._atom_types[angle_atoms[:, [1, 0]]]), self._nonsymmetric_pairing_function(self._atom_types[angle_atoms[:, [1, 2]]]))))
        return torch.empty_like(angle_atoms)

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
        self._atom_types_in_angle = torch.tensor([(self._name_to_atom_type[self._unique_angle_type_names[i][0]], self._name_to_atom_type[self._unique_angle_type_names[i][1]], self._name_to_atom_type[self._unique_angle_type_names[i][2]]) for i in torch.arange(len(self._unique_angle_type_names))], device=self.device)
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
        pair_types = torch.zeros(pairs.size(0), dtype=torch.long, device=self.device)
        i_less_than_j_indices = torch.where(pairs[:, 0] < pairs[:, 1])[0].to(self.device)
        other_indices = torch.where(pairs[:, 0] >= pairs[:, 1])[0].to(self.device)
        pair_types[i_less_than_j_indices] = torch.square(pairs[i_less_than_j_indices][:, 1]) + pairs[i_less_than_j_indices][:, 0]
        pair_types[other_indices] = torch.square(pairs[other_indices][:, 0]) + pairs[other_indices][:, 0] + pairs[other_indices][:, 1]
        return pair_types

    def _flatten_raw_parameter_dicts_to_arrays(self, raw_atomic_params: Dict[str, torch.Tensor], raw_pair_params: Dict[Tuple[str, str], torch.Tensor], raw_pair_pair_params: Dict[Tuple[Tuple[str, str], Tuple[str, str]], torch.Tensor], raw_pair_angle_params: Dict[Tuple[Tuple[str, str], Tuple[str, str, str]], torch.Tensor], raw_angle_params: Dict[Tuple[str, str, str], torch.Tensor]):
        # Atom Types #
        n_types = len(self._unique_atom_type_names)
        for param_key in raw_atomic_params[self._unique_atom_type_names[0]]:
            self._atomic_param_arrays[param_key] = torch.zeros_like(raw_atomic_params[self._unique_atom_type_names[0]][param_key], device=self.device)
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
        # puts them in the range 0:N_pair_types, etc. The other option is to turn these dense arrays
        # into sparse arrays whose only valid indices are the pair types, angle types, etc.

        # Pair Types #
        for i in range(len(self._unique_pair_type_names)):
            if self._unique_pair_type_names[i] in raw_pair_params:
                for param_key in raw_pair_params[self._unique_pair_type_names[i]]:
                    self._pair_param_arrays[param_key] = torch.zeros(torch.max(self._unique_pair_types) + 1, device=self.device)
                #break
        
        # Build matrix indexable by two Pair Types #
        for pair_tuple in raw_pair_pair_params.keys(): # Loop over names
            for param_key in raw_pair_pair_params[pair_tuple].keys():
                self._pair_pair_param_arrays[param_key] = torch.zeros(torch.max(self._unique_pair_types) + 1, torch.max(self._unique_pair_types) + 1, device=self.device)

        # Build matrix indexable by a Pair Type and Angle Type #
        for pair_angle_tuple in raw_pair_angle_params.keys(): # Loop over names
            for param_key in raw_pair_angle_params[pair_angle_tuple].keys():
                self._pair_angle_param_arrays[param_key] = torch.zeros(torch.max(self._unique_pair_types) + 1, torch.max(self._unique_angle_types) + 1, device=self.device)

        # Angle Types #
        for i in range(len(self._unique_angle_type_names)):
            if self._unique_angle_type_names[i] in raw_angle_params:
                for param_key in raw_angle_params[self._unique_angle_type_names[i]]:
                    self._angle_param_arrays[param_key] = torch.zeros(torch.max(self._unique_angle_types) + 1, device=self.device)
                break # <------ I think this is actually a bug when there are more angle types than HOH???

    def _build_atomic_parameter_arrays(self, raw_atomic_params: Dict[str, torch.Tensor]):
        for param_key in self._atomic_param_arrays.keys():
            for i in torch.arange(len(self._unique_atom_type_names)):
                self._atomic_param_arrays[param_key][i] = raw_atomic_params[self._unique_atom_type_names[i]][param_key]

    def _build_pair_parameter_arrays(self, raw_pair_params: Dict[Tuple[str, str], torch.Tensor]):
        #print(raw_pair_params[("O_water", "H_water")])
        #print(self._pair_param_arrays.keys())
        for param_key in self._pair_param_arrays.keys():
            for i in torch.arange(len(self._unique_pair_type_names)):
                if self._unique_pair_type_names[i] in raw_pair_params and param_key in raw_pair_params[self._unique_pair_type_names[i]]:
                    self._pair_param_arrays[param_key][self._name_to_pair_type[self._unique_pair_type_names[i]]] = raw_pair_params[self._unique_pair_type_names[i]][param_key][0]
    
    def _build_pair_pair_parameter_arrays(self, raw_pair_pair_params: Dict[Tuple[Tuple[str, str], Tuple[str, str]], torch.Tensor]):
        for param_key in self._pair_pair_param_arrays.keys():
            for i in torch.arange(len(self._unique_pair_type_names)):
                for j in torch.arange(len(self._unique_pair_type_names)):
                    if (self._unique_pair_type_names[i], self._unique_pair_type_names[j]) in raw_pair_pair_params:
                        self._pair_pair_param_arrays[param_key][self._name_to_pair_type[self._unique_pair_type_names[i]], self._name_to_pair_type[self._unique_pair_type_names[j]]] = raw_pair_pair_params[(self._unique_pair_type_names[i], self._unique_pair_type_names[j])][param_key][0]

    def _build_pair_angle_parameter_arrays(self, raw_pair_angle_params: Dict[Tuple[Tuple[str, str], Tuple[str, str, str]], torch.Tensor]):
        for param_key in self._pair_angle_param_arrays.keys():
            for i in torch.arange(len(self._unique_pair_type_names)):
                for j in torch.arange(len(self._unique_angle_type_names)):
                    if (self._unique_pair_type_names[i], self._unique_angle_type_names[j]) in raw_pair_angle_params:
                        self._pair_angle_param_arrays[param_key][self._name_to_pair_type[self._unique_pair_type_names[i]], self._name_to_angle_type[self._unique_angle_type_names[j]]] = raw_pair_angle_params[(self._unique_pair_type_names[i], self._unique_angle_type_names[j])][param_key][0]

    def _build_angle_parameter_arrays(self, raw_angle_params: Dict[Tuple[str, str, str], torch.Tensor]):
        for param_key in self._angle_param_arrays.keys():
            for i in torch.arange(len(self._unique_angle_type_names)):
                if self._unique_angle_type_names[i] in raw_angle_params:
                    self._angle_param_arrays[param_key][self._name_to_angle_type[self._unique_angle_type_names[i]]] = raw_angle_params[self._unique_angle_type_names[i]][param_key][0]

    def get_atomic_parameters(self, name: str):
        return self._atomic_param_arrays[name][self._atom_types].squeeze_()
    
    def get_pair_parameters(self, name: str, pairs_p: torch.Tensor):
        return self._pair_param_arrays[name][self._pair_types[pairs_p]]

    def get_pair_parameters_with_optional_combination_rule(self, name: str, pairs_p: torch.Tensor, pairs_a: torch.Tensor, combination_rule=torch.sqrt):

        pair_types = self._pair_types[pairs_p]
        pair_params = torch.ones_like(pair_types, device=self.device, dtype=torch.get_default_dtype()) * -123456789.0
        # ^^^ I am guessing there will not be any force field with the parameter -123456789
        # but if there is, then the code will break. Both 1 and 0 are quite likely to be actual
        # pair parameter values.
        if name in self._pair_param_arrays.keys():
            pair_params = self._pair_param_arrays[name][pair_types]
        
        mask = torch.where(pair_params != -123456789.0)[0].flatten()
        pair_specific_params = pair_params[mask]
        if name in self._atomic_param_arrays.keys() and mask.size() != pair_params.size():
            atomic_params_1 = self._atomic_param_arrays[name][self._atom_types].squeeze_()[pairs_a[:,0]]
            atomic_params_2 = self._atomic_param_arrays[name][self._atom_types].squeeze_()[pairs_a[:,1]]
            pair_params = combination_rule(atomic_params_1 * atomic_params_2)
        pair_params[mask] = pair_specific_params
        return pair_params

    def get_angle_parameters(self, name: str, angle_atoms_a: torch.Tensor):
        return self._angle_param_arrays[name][self._get_angle_types_from_angle_atoms(angle_atoms_a)]

    def get_pair_pair_parameters(self, name: str, pairs_1_p: torch.Tensor, pairs_2_p: torch.Tensor):
        return self._pair_pair_param_arrays[name][self._pair_types[pairs_1_p], self._pair_types[pairs_2_p]]
    
    def get_pair_angle_parameters(self, name: str, angle_pairs_p: torch.Tensor, angle_atoms_a: torch.Tensor):
        # NOTE(JOE): _get_angle_types_from_angle_atoms will return the type of each angle. Since there are two
        # bonds in each angle (which may in general be different), we have to repeat the angle types twice.
        # angle_pairs_p comes from the topology object as two columns of pair indices hence the flattening.
        return self._pair_angle_param_arrays[name][self._pair_types[angle_pairs_p.T.flatten()], self._get_angle_types_from_angle_atoms(angle_atoms_a).repeat_interleave(2)]
