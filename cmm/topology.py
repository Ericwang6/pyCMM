import torch

class Topology:
    def __init__(self):
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.improper_dihedrals = []

    def add_bond(self, atom_idx_1: int, atom_idx_2: int):
        # Add bond to the topology
        pass

    def add_angle(self, atom_idx_1: int, atom_idx_2: int, atom_idx_3: int):
        pass

    def add_dihedral(self, atom_idx_1: int, atom_idx_2: int, atom_idx_3: int, atom_idx_4: int):
        raise NotImplementedError
    
    def add_improper_dihedral(self, atom_idx_1: int, atom_idx_2: int, atom_idx_3: int, atom_idx_4: int):
        raise NotImplementedError