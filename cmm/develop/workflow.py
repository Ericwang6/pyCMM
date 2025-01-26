import os, sys, glob, shutil
import subprocess
import logging
from pathlib import Path
import numpy as np
import parmed
from .qchem import QChemReader, QChemTask
from .gdma import GDMATask, PoleditGDMATask
from .base import Molecule
from .xtb import XTBMDTask
from .xyz import XYZReader
from .multiwfn import MultiwfnTask


def run_command(cmd, wdir):
    logging.info(f'The following command is running at {wdir}')
    logging.info(cmd)
    os.system(f'cd {wdir} && {cmd}')


input_pdb = ""
charge = 0
wdir = ""

wdir = Path(wdir)
wdir.mkdir(exist_ok=True)

shutil.copy(input_pdb, wdir)

struct = parmed.load_file(input_pdb)
atoms = [parmed.periodic_table.Element[at.element] for at in struct.atoms]
positions = struct.coordinates

molecule = Molecule(atoms, positions, charge)

# Optimize
opt_dir = wdir / '01.opt'
opt_config = {'jobtype': 'opt', 'num_cores': 128}
opt_task = QChemTask(opt_dir, molecule, opt_config)
opt_task.run()

_, opt_coord, dipo = QChemReader.read_out(opt_task.stdout)
molecule.coords = opt_coord
molecule.dipo = dipo

# GDMA
gdma_dir = wdir / '02.gdma'
gdma_task = GDMATask(wdir, opt_task.wdir / f'{opt_task.name}.fchk')
gdma_task.run()

poledit = PoleditGDMATask(gdma_dir, gdma_task.stdout)
poledit.run()

mwfn_task = MultiwfnTask(gdma_dir, opt_task.wdir / f'{opt_task.name}.fchk')
mwfn_task.run()

# Sampling xtb-md
xtb_dir = wdir / '03.xtbmd'
xtb_config = {}
xtb_task = XTBMDTask(wdir, molecule, xtb_config)
xtb_task.run()

# QChem
qchem_dir = wdir / '04.qchem'
qchem_dir.mkdir(exist_ok=True)

xtb_molecules = XYZReader(xtb_task / f'{xtb_task.name}.trj').read()
for i, mol in enumerate(xtb_molecules):
    mol.charge = molecule.charge
    task = QChemTask(qchem_dir / str(i), mol, config={'jobtype': 'force', 'num_cores': 128}, name='qchem')
    task.run()

