import os
import logging
from pathlib import Path
from typing import Optional
from .base import Task


class MultiwfnTask(Task):

    multiwfn_in = '\n'.join(['7', '13', '6', '3', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0', 'q', '1', '1', '0', 'y', '0', '0', 'q'])

    def __init__(self, wdir: os.PathLike, fchk_file: os.PathLike, name: str = 'multiwfn', logger: Optional[logging.Logger] = None):
        super().__init__(name, wdir, logger)
        self.fchk_file = Path(fchk_file).resolve()
    
    def prep(self):
        with open(self.stdin, 'w') as f:
            f.write(self.multiwfn_in)
        self.cmd = f'Multiwfn {self.fchk_file}'
        return self.cmd