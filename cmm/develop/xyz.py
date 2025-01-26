import os
from pathlib import Path
from typing import List, Union, TextIO
import numpy as np
from .base import Molecule


class XYZWriter:
    def __init__(self, file: os.PathLike):
        self.file = file
    
    def write(self, molecule: Molecule):
        self.write_file(molecule.atoms, molecule.coords, molecule.name, self.file)

    @staticmethod
    def write_file(atoms: List[str], coords: np.ndarray, comment: str = '', file: Union[os.PathLike, TextIO, None] = None):
        lines = [str(len(atoms)), comment]
        for at, crd in zip(atoms, coords):
            lines.append(f"{at:>3} {crd[0]:>13.7f} {crd[1]:>13.7f} {crd[2]:>13.7f}")
        inp = '\n'.join(lines)

        if isinstance(file, str) or isinstance(file, Path):
            with open(file, 'w') as f:
                f.write(inp)
        elif isinstance(file, TextIO):
            file.write(inp)
        return inp


class XYZReader:
    def __init__(self, file: os.PathLike):
        self.file = file
    
    def read(self):
        atoms_list, coords_list = self.read_file(self.file)
        molecules = []
        for atoms, coord in zip(atoms_list, coords_list):
            molecules.append(Molecule(atoms, coord))
        return molecules

    @staticmethod
    def read_file(fname: os.PathLike):
        atoms_list = []
        coords_list = []
        with open(fname) as f:
            while True:
                start = f.readline()
                if not start:
                    break
                natoms = int(start.strip())
                comment = f.readline()
                atoms, coords = [], []
                for _ in range(natoms):
                    line = f.readline().strip().split()
                    atoms.append(line[0])
                    coords.append(list(map(float, line[1:])))
                coords = np.array(coords)
                atoms_list.append(atoms)
                coords_list.append(coords)
        return atoms_list, coords_list