import torch
from typing import List

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

if __name__ == "__main__":
    grid = torch.linspace(-10.0, 10.0, 10)
    positions = torch.cartesian_prod(grid, grid, grid)
    write_xyz("temp.xyz", ["He" for _ in range(positions.size()[0])], positions)