# NOTE(JOE): Can also imagine defining an AtomClass object which just populates
# to the AtomTypes so that AtomTypes can be shared between atoms in the same AtomClass.

class AtomTypes:
    """
    This class defines the type of atoms. The type of a particular atom is specified
    by a case-insensitive string such as "C_sp3" or "O_water".
    
    Internally, an atom type is a unique integer (basically an enum)
    which can be used to associate particular atoms in a system with the appropriate
    parameters and other data. This integer is referred to as the id.
    
    This class provides utilities to convert between the string representation
    and the atom type index and to specify new atom types.
    Force fields are responsible for implementing anything that
    needs to be done with the atom types as well as defining new atom types as needed.
    """
    def __init__(self):
        # All hard-coded atom types must begin with the
        # element followed by "_" then anything can follow.
        self.type_to_id = {
            "O_water": 1,
            "H_water": 2
        }
        self.id_to_type = {} # maps type_id to type_name (reverse map of self.type_to_id)
        self.type_to_element = {} # maps the type_name to element

        for key, value in self.type_to_id.items():
            self.id_to_type[value] = key
        
        for key, value in self.type_to_id.items():
            self.type_to_element[key] = key.split("_")[0].title()

    def add_atom_type(self, type_name: str, element: str):
        if type_name not in self.type_to_id:
            new_id = len(self.type_to_id) + 1 # Since the lowest type_id is 1, not 0.
            self.type_to_id[type_name] = new_id
            self.id_to_type[new_id] = type_name
            self.type_to_element[type_name] = element.title()

    def get_type_id(self, type_name: str):
        return self.type_to_id[type_name]

    def get_type_name(self, type_id: int):
        return self.id_to_type[type_id]

    def get_element_by_type_id(self, type_id: int):
        return self.type_to_element[self.id_to_type[type_id]]
    
    def get_element_by_type_name(self, type_name: str):
        return self.type_to_element[type_name]

if __name__ == "__main__":
    atom_types = AtomTypes()
    assert atom_types.get_type_id("O_water") == 1
    assert atom_types.get_type_id("H_water") == 2
    atom_types.add_atom_type("N_NH3+", "N")
    new_id = atom_types.get_type_id("N_NH3+")
    assert atom_types.get_element_by_type_id(new_id) == "N"
    assert atom_types.get_element_by_type_name(atom_types.id_to_type[new_id]) == "N"
