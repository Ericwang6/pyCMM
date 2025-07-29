import os
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from .base import Task


class MultiwfnTask(Task):

    multiwfn_in = '\n'.join([])

    def __init__(self, wdir: os.PathLike, fchk_file: os.PathLike, name: str = 'multiwfn', logger: Optional[logging.Logger] = None):
        super().__init__(name, wdir, logger)
        self.fchk_file = Path(fchk_file).resolve()
    
    def prep(self):
        with open(self.stdin, 'w') as f:
            f.write(self.multiwfn_in)
        self.cmd = f'Multiwfn {self.fchk_file}'
        return self.cmd


class MultiwfnReader:

    @staticmethod
    def read_chg_file(fname: os.PathLike):
        atoms, coord, chgs = [], [], []
        with open(fname) as f:
            for line in f:
                content = line.strip().split()
                atoms.append(content[1])
                coord.append(list(map(float, content[1:4])))
                chgs.append(float(content[4]))
        coord = np.array(coord)
        chgs = np.array(chgs)
        return atoms, coord, chgs
    
    @staticmethod
    def read_esp_file(fname: os.PathLike):
        coords, esp = [], []
        with open(fname) as f:
            natoms = int(f.readline().strip())
            for i in range(natoms):
                line = list(map(float, f.readline().strip().split()))
                coords.append(line[:3])
                esp.append(line[3])
        coords = np.array(coords)
        esp = np.array(esp)
        return coords, esp