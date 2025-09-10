import os
from pathlib import Path
from typing import Dict, Any, Union, TextIO, Optional
import logging
from .base import Task, Molecule
from .xyz import XYZWriter


class XTBWriter:

    @staticmethod
    def write_inp(tag: str, config: Dict[str, Any], file: Union[os.PathLike, TextIO, None] = None):
        maxlen = max([len(key) for key in config.keys()])
        tmpl = "   {:<" + str(maxlen) + "}  =  {}"
        lines = [f'${tag}']
        for key, value in config.items():
            if isinstance(value, bool): 
                value = str(value).lower()
            line = tmpl.format(key.lower(), value)
            lines.append(line)
        lines.append('$end')
        inp = '\n'.join(lines)

        if isinstance(file, str) or isinstance(file, Path):
            with open(file, 'w') as f:
                f.write(inp)
        elif isinstance(file, TextIO):
            file.write(inp)
        
        return inp


class XTBMDTask(Task):

    default_config = {
        "temp": 500.0,
        "dump": 500.0,
        "step": 2.0,
        "shake": 0,
        "time": 100.0
    }

    def __init__(self, wdir: os.PathLike, molecule: Molecule, config: Dict[str, Any] = dict(), name: str = 'xtbmd', logger: Optional[logging.Logger] = None):
        super().__init__(name, wdir, logger)
        self.molecule = molecule
        self.config = self.default_config.copy()
        self.config.update(config)
        self.xyz = self.wdir / f'{self.name}.xyz'
        self.input = self.stdin
        self.stdin = None

    def prep(self):
        XTBWriter.write_inp('md', self.config, self.input)
        XYZWriter(self.xyz).write(self.molecule)
        self.cmd = f'xtb {self.xyz} --input {self.input} --md --chrg {self.molecule.charge} --uhf 0 --ceasefiles'
        return self.cmd

