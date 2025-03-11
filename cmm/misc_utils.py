import torch
from typing import List
import numpy as np

def write_xyz(outfile: str, labels: List[str], coords: torch.Tensor) -> None:
    """
    Writes a plain xyz file.
    
    Args:
        outfile (str): Name of output file.
        labels (List[str]): Element labels of all atoms (N,).
        coords (torch.Tensor): Positions of all atoms (N, 3).
    """

    natoms = coords.size()[0]
    assert natoms == len(labels), "Number of atom labels and coordinates do not match."
    with open(outfile, "w") as f:
        header = str(natoms) + "\n\n"
        f.write(header)
        for i in range(natoms):
            vec = coords[i, :]
            line = labels[i] + " " + str(vec[0].item()) + " " + str(vec[1].item()) + " " + str(vec[2].item()) + "\n"
            f.write(line)

def read_xyz_tinker(infile: str):
    """
    Reads an xyz formatted file following the tinker convention.
    Specifically, the first column is the atom number, second is
    the atom labels, then xyz coordinates, then the integer atom type,
    followed by the atom numbers to which this atom is connected.
    An atom need not be connected to any other atoms.
    """
    atom_numbers = []
    atom_labels = []
    coords = []
    atom_types = []

    with open(infile, "r") as f:
        lines = f.readlines()
        header = lines.pop(0)
        
        # The point here is that the connectivity should be symmetric
        # when transposed. We only actually need the bonds in one
        # direction (i<j by choice), but we keep track of both when parsing 
        # to make sure that the topology is well-defined.
        natoms = int(header.split()[0])
        bonds_start_i_less_than_j = []
        bonds_end_i_less_than_j = []
        bonds_start_i_greater_than_j = []
        bonds_end_i_greater_than_j = []
        for line in lines:
            split_line = line.split()
            atom_number = int(split_line[0])
            atom_numbers.append(atom_number)
            atom_labels.append(str(split_line[1]))
            coords.append(np.array([split_line[2], split_line[3], split_line[4]], dtype=np.float64))
            atom_types.append(int(split_line[5]))
            if len(split_line) > 6: # The rest of the line is the bonding info
                bonds = split_line[6:]
                for bond_end in bonds:
                    bond_end = int(bond_end)
                    if bond_end > atom_number:
                        bonds_start_i_less_than_j.append(atom_number)
                        bonds_end_i_less_than_j.append(bond_end)
                    elif bond_end < atom_number:
                        bonds_start_i_greater_than_j.append(atom_number)
                        bonds_end_i_greater_than_j.append(bond_end)
                    else:
                        raise ValueError("Tinker formatted xyz file indicates an atom is bonded to itself. Please ensure the input is correct.")
    
    # TODO: Could add option to ignore this check since one could realistically omit connectivity
    # info from the file for atoms which are connected to atoms of a smaller index.
    if bonds_start_i_less_than_j != bonds_end_i_greater_than_j or bonds_start_i_greater_than_j != bonds_end_i_less_than_j:
        raise ValueError("Bond connectivity in tinker xyz file is not symmetric. Please ensure input is correct.")
    
    # Subtract 1 from bond arrays because tinker xyz specifies the first atom starting from 1.
    return atom_labels, np.array(atom_types, dtype=np.int64), np.vstack(coords), np.array([np.array(bonds_start_i_less_than_j) - 1, np.array(bonds_end_i_less_than_j) - 1])



if __name__ == "__main__":
    grid = torch.linspace(-10.0, 10.0, 10)
    positions = torch.cartesian_prod(grid, grid, grid)
    write_xyz("temp.xyz", ["He" for _ in range(positions.size()[0])], positions)