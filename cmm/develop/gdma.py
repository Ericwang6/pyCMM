import os
import logging
from pathlib import Path
from typing import Optional
from .base import Task


def write_gdma_input(fchk_file: os.PathLike, name: str = 'gdma', file: os.PathLike = ''):

    gdma_inp = f'''Title "{name} input"
File {fchk_file}

Angstrom
Multipoles
  switch 4
  Limit 2
  Limit 2 H 
  Radius H 0.325
  Punch {name}.punch
Start

Finish'''
    
    if file:
        with open(file, 'w') as f:
            f.write(gdma_inp)


class GDMATask(Task):
    def __init__(self, wdir: os.PathLike, fchk_file: os.PathLike, name: str = 'gdma', logger: Optional[logging.Logger] = None):
        super().__init__(name, wdir, logger)
        self.fchk_file = Path(fchk_file).resolve()
    
    def prep(self):
        self.cmd = 'gdma'
        write_gdma_input(self.fchk_file, self.name, self.stdin)
        return self.cmd
    

class PoleditGDMATask(Task):

    poledit_in = '''

A

1
Y

Y'''

    def __init__(self, wdir: os.PathLike, gdma_out: os.PathLike, name: str = 'poledit', logger: Optional[logging.Logger] = None):
        super().__init__(name, wdir, logger)
        self.gdma_out = Path(gdma_out).resolve()
    
    def prep(self):
        self.cmd = f'poledit 1 {self.gdma_out}'
        with open(self.stdin, 'w') as f:
            f.write(self.poledit_in)
        return self.cmd