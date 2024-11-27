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
    with open(infile, "r") as f:
        # TODO: Parse the header which could contain box info in principle.
        lines = f.readlines()[1:]

        for line in lines:
            split_line = line.split()
            atom_numbers.append(int(split_line[0]))
            atom_labels.append(str(split_line[1]))
            coords.append(np.array([split_line[2], split_line[3], split_line[4]], dtype=np.float64))
            # TODO: Actually parse the connectivity.
    return atom_labels, np.vstack(coords)



if __name__ == "__main__":
    grid = torch.linspace(-10.0, 10.0, 10)
    positions = torch.cartesian_prod(grid, grid, grid)
    write_xyz("temp.xyz", ["He" for _ in range(positions.size()[0])], positions)