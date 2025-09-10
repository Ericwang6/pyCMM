import os, uuid, logging, shutil, subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Molecule:
    atoms: List[str]
    coords: np.ndarray
    charge: int = 0
    mult: int = 1
    dipo: np.ndarray = None
    quad: np.ndarray = None
    forces: np.ndarray = None
    energy: np.ndarray = None
    hessian: np.ndarray = None
    polarizability: np.ndarray = None
    name: str = "Molecule"


def init_logger(logname: Optional[os.PathLike] = None) -> logging.Logger:
    # logging
    logger = logging.getLogger(str(uuid.uuid4()))
    logger.propagate = False
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # file
    if logname is not None:
        handler = logging.FileHandler(str(logname))
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


Logger = init_logger()


class Task(ABC):
    def __init__(self, name: str, wdir: os.PathLike, logger: Optional[logging.Logger] = None):
        self.name = name
        self.wdir = Path(wdir).resolve()
        self.wdir.mkdir(exist_ok=True)
        
        self.logger = Logger if logger is None else logger
        self.stdin = self.wdir / f'{self.name}.in'
        self.stdout = self.wdir / f'{self.name}.out'
        self.stderr = self.wdir / f'{self.name}.err'
        self.cmd = ''
        
    # @abstractmethod
    def prep(self) -> str:
        return

    def main(self):
        if not self.cmd:
            raise ValueError('Please set `cmd` attribute in the .prep() method')
        args = self.cmd.split() if isinstance(self.cmd, str) else self.cmd
        self.logger.info(f'In working directory: {self.wdir}')
        self.logger.info('The following command is executed: "{}"'.format(" ".join(args)))

        stdin = open(self.stdin) if self.stdin else None
        stdout = open(self.stdout, 'w')
        stderr = open(self.stderr, 'w')
        self.sub = subprocess.Popen(
            args=args,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            cwd=str(self.wdir)
        )
        self.sub.communicate()
        if self.stdin:
            stdin.close()
        stdout.close()
        stderr.close()
        if self.sub.returncode != 0:
            raise CommandExecuteError(args, self.stderr)

    def after(self):
        return 

    def run(self):
        tag = self.wdir / 'running.tag'
        tag.touch()
        try:
            self.prep()
            self.main()
            self.after()
        except Exception as e:
            shutil.move(tag, tag.with_name('error.tag'))
            raise e
        shutil.move(tag, tag.with_name('done.tag'))


class CommandExecuteError(Exception):
    """
    Exception for command line exec error
    """
    def __init__(self, cmd, f_err):
        if isinstance(cmd, list):
            _cmd = ' '.join(cmd)
        self._errmsg = f'Command {_cmd} failed. Please check {f_err} for more details.'
    
    def __str__(self):
        return self._errmsg
    
    def __repr__(self):
        return self._errmsg



if __name__ == '__main__':
    task = Task('test', '.')
    task.cmd = 'ls -lh'
    task.stdin = None
    task.run()