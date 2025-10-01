from typing import List, Dict, Any, Optional, TextIO, Union, Literal
from collections import defaultdict
import os
import numpy as np
import parmed
import logging
from pathlib import Path
import warnings

from .base import Molecule, Task


def to_pdb(coords, symbols, fname, resname="UNK"):
    struct = parmed.Structure()
    res = parmed.Residue(name=resname)
    struct.residues.append(res)
    count = {}
    for coord, symb in zip(coords, symbols):
        cnt = count.get(symb, 0) + 1
        count[symb] = cnt
        atname = f"{symb}{cnt}"
        atom = parmed.Atom(
            atomic_number=parmed.periodic_table.AtomicNum[symb],
            name=atname
        )
        atom.xx, atom.xy, atom.xz = tuple(map(float, coord))
        struct.add_atom(atom, resname, -1)
    struct.save(fname, overwrite=True, use_hetatoms=False)


class QChemWriter:

    default_job_config = {
        "METHOD":  "wB97X-V",
        "BASIS": "def2-QZVPPD",
        "XC_GRID": "000099000590",
        "NL_GRID": 1,
        "MEM_TOTAL": 64000,
        "MEM_STATIC": 10000,
        "IQMOL_FCHK": True,
    }

    default_eda_config = {
        "JOBTYPE": "eda",
        "EDA2": 1,
        "METHOD": "wB97X-V",
        "BASIS": "def2-QZVPPD",
        "SCF_CONVERGENCE": 8,
        "THRESH": 14,
        "XC_GRID": "000099000590",
        "NL_GRID": 1,
        "SYMMETRY": False,
        "EDA_BSSE": False,
        "FD_MAT_VEC_PROD": False,
        "MEM_TOTAL": 64000,
        "MEM_STATIC": 10000,
        "SCF_PRINT_FRGM": True
    }

    def __init__(self, file: os.PathLike, jobtype: str = 'sp', config: Dict[str, Any] = dict()):
        self.config = {key.upper(): value for key, value in config.items()}
        self.jobtype = jobtype
        if ('JOBTYPE' in self.config) and self.config['JOBTYPE'] != self.jobtype:
            warnings.warn(f'Inconsisent job type. Jobtype {jobtype} will be used')
        self.config['JOBTYPE'] = self.jobtype

        if self.jobtype == 'eda':
            config = self.default_eda_config.copy()
        else:
            config = self.default_job_config.copy()
        config.update(self.config)
        self.config = config

        self.file = open(file, 'w')
    
    def close(self):
        self.file.close()

    def write_opt(self, molecule: Molecule):
        self.write_molecule(
            molecule.atoms, 
            molecule.coords, 
            molecule.charge, 
            molecule.mult,
            self.file
        )
        opt_config = self.config.copy()
        opt_config['JOBTYPE'] = 'opt'
        opt_config['IQMOL_FCHK'] = False
        self.file.write('\n\n')
        self.write_rem(opt_config, self.file)
        self.file.write('\n\n@@@@\n\n')
        self.file.write('$molecule\nread\n$end\n\n')

        hessian_config = self.config.copy()
        hessian_config['JOBTYPE'] = 'freq'
        hessian_config['IQMOL_FCHK'] = True
        self.write_rem(hessian_config, self.file)
        self.file.write('\n\n')
        self.file.close()

    def write_simple(self, molecule: Molecule):
        self.write_molecule(
            molecule.atoms,
            molecule.coords,
            molecule.charge,
            molecule.mult,
            self.file
        )
        self.file.write('\n\n')
        self.write_rem(self.config, self.file)
        self.file.write('\n')
        self.close()
    
    def write_eda(self, molecule: Molecule):
        raise NotImplementedError()

    def write(self, molecule: Molecule):
        if self.jobtype == 'opt':
            self.write_opt(molecule)
        elif self.jobtype == 'eda':
            self.write_eda(molecule)
        else:
            self.write_simple(molecule)

    @staticmethod
    def write_rem(config: Dict[str, Any], file: Optional[TextIO] = None):
        maxlen = max([len(key) for key in config.keys()])
        tmpl = "   {:<" + str(maxlen) + "}  =  {}\n"
        lines = ['$rem\n']
        for key, value in config.items():
            if isinstance(value, bool): 
                value = str(value).lower()
            line = tmpl.format(key.upper(), value)
            lines.append(line)
        lines.append('$end')
        remstr = ''.join(lines)
        if file:
            file.write(remstr)
        return remstr
    
    @staticmethod
    def write_molecule(atoms: List[str], coords: np.ndarray, charge: int, mult: int, file: Optional[TextIO] = None):
        assert coords.shape == (len(atoms), 3), "Coordinates do not match with the atoms"
        lines = ["$molecule\n", f'{charge} {mult}\n']
        for at, crd in zip(atoms, coords):
            lines.append(f"{at:>3} {crd[0]:>13.7f} {crd[1]:>13.7f} {crd[2]:>13.7f}")
        lines.append('$end')
        molstr = '\n'.join(lines)
        if file:
            file.write(molstr)
        return molstr
    
    @staticmethod
    def write_molecule_list(atoms: List[List[str]], coords: List[np.ndarray], charges: List[int], mults: List[int], total_mult: int = 1, names: Optional[List[str]] = None, file: Optional[TextIO] = None):
        total_charge = sum(charges)
        lines = ["$molecule\n", f'{total_charge} {total_mult}\n']
        names = ["" for _ in range(len(atoms))] if names is None else names
        for i in range(len(atoms)):
            lines.append(f"-- {names[i]}\n{charges[i]} {mults[i]}\n")
            for at, crd in zip(atoms[i], coords[i]):
                lines.append(f"{at:>3} {crd[0]:>13.7f} {crd[1]:>13.7f} {crd[2]:>13.7f}\n")
        lines.append('$end')
        molstr = ''.join(lines)
        if file:
            file.write(molstr)
        return molstr
    

def _parse_force_line(line: str):
    return [float(line[5:17]), float(line[17:29]), float(line[29:41])]

def _parse_force_block(f, natoms: int):
    results = []
    for i in range(natoms):
        results.append(_parse_force_line(f.readline()))
    return np.array(results)

def _parse_molecule_block_with_fragments(f):
    total_charge, total_mult = tuple(map(int, f.readline().strip().split()))
    charges = []
    mults = []
    atoms = []
    coords = []
    while True:
        line = f.readline()
        if line.startswith('$end'):
            break
        elif line.startswith('-'):
            charge, mult = tuple(map(int, f.readline().strip().split()))
            charges.append(charge)
            mults.append(mult)
            atoms.append([])
            coords.append([])
        else:
            content = line.strip().split()
            atoms[-1].append(content[0])
            coords[-1].append(list(map(float, content[1:])))
    return atoms, coords, charges, mults, total_charge, total_mult


class QChemReader:
    def __init__(self):
        pass

    @staticmethod
    def read_eda_out(out: os.PathLike, unit: str = 'kcal'):
        res = {
            "TOTAL": None, "PREPARATION": None,
            "ELEC": None, "CLS_ELEC": None,
            "PAULI": None, "MOD_PAULI": None, "DISP": None,
            "FROZEN": None, "POLARIZATION": None, "CHARGE_TRANSFER": None,
        }
        _read = False
        _read_coords = False
        old_version = None
        coords_lines = []

        with open(out) as f:
            for line in f:
                line = line.strip()

                if (old_version is None) and line.startswith("Q-Chem"):
                    if int(line.split()[1].split('.')[0]) < 6:
                        old_version = True
                    else:
                        old_version = False
                
                if (not old_version) and line.startswith("Decomposition of frozen interaction energy"):
                    _read = True
                    continue
                if (old_version) and line.startswith("Initial Wavefunction Decomposition"):
                    _read = True
                    continue
                if (not _read_coords) and len(coords_lines) == 0 and line.startswith('$molecule'):
                    _read_coords = True
                    continue
                if _read_coords and line.startswith('$end'):
                    _read_coords = False
                    continue

                if _read_coords:
                    coords_lines.append(line)

                if _read:
                    content = line.split()
                    if len(content) == 0:
                        continue
                    if content[0] == "E_elec":
                        res['ELEC'] = float(content[-1])
                    elif content[0] == "E_pauli":
                        res['PAULI'] = float(content[-1])
                    elif content[0] == "E_disp":
                        res["DISP"] = float(content[-1])
                    elif content[0] == "E_cls_elec":
                        res['CLS_ELEC'] = float(content[-1])
                    elif content[0] == "E_mod_pauli" or content[0] == '[E_mod_pauli':
                        res['MOD_PAULI'] = float(content[5])
                    elif content[0] == "PREPARATION":
                        res["PREPARATION"] = float(content[-1])
                    elif content[0] == "FROZEN":
                        res["FROZEN"] = float(content[1])
                        if old_version:
                            res['FROZEN'] += res['DISP']
                    elif content[0] == "POLARIZATION":
                        res['POLARIZATION'] = float(content[-1])
                    elif content[0] == "CHARGE":
                        res['CHARGE_TRANSFER'] = float(content[-1])
                    elif content[0] == "TOTAL":
                        res["TOTAL"] = float(content[1])
                        _read = False
                        break
        
        # process coordinate lines
        atoms = []
        coords = []
        total_charge, total_mult = tuple(map(int, coords_lines[0].split()))
        charges, mults = [], []
        for line in coords_lines[1:]:
            if not line:
                continue
            
            if line.startswith('--'):
                _new = True
                continue
            
            if _new:
                charge, mult = tuple(map(int, line.split()))
                charges.append(charge)
                mults.append(mult)
                atoms.append([])
                coords.append([])
                _new = False
                continue

            content = line.split()
            atoms[-1].append(content[0])
            coords[-1].append(list(map(float, content[1:])))
        
        coords = [np.array(coord) for coord in coords]

        for key in res:
            if res[key] is None:
                raise RuntimeError(f"Fail to parse {out}")

        assert abs(res["ELEC"] + res["PAULI"] - res["CLS_ELEC"] - res["MOD_PAULI"]) < 2e-4
        frozen_error = abs(res["FROZEN"] - res["CLS_ELEC"] - res['MOD_PAULI'] - res['DISP'])
        assert frozen_error < 2e-4, f"{frozen_error} too large"
        
        total_error = abs(res['TOTAL'] - res['PREPARATION'] - res['FROZEN'] - res['CHARGE_TRANSFER'] - res['POLARIZATION'])
        assert total_error < 5e-4, f"{total_error} too large"
        
        if unit == "kcal":
            for key in res.keys():
                res[key] /= 4.184

        return atoms, coords, charges, res
    
    @staticmethod
    def read_out(out: Union[TextIO, os.PathLike]):
        _read_coord = False
        polarizability = None
        with open(out) as f:
            for line in f:
                if line.strip().startswith("Standard Nuclear Orientation"):
                    _read_coord = True
                    atoms = []
                    coords = []
                    f.readline()
                    f.readline()
                    continue

                if _read_coord and line.strip().startswith('-'):
                    _read_coord = False
                    continue

                if _read_coord:
                    content = line.strip().split()
                    coords.append([float(content[-3]), float(content[-2]), float(content[-1])])
                    atoms.append(content[1])
                
                if line.strip().startswith('Dipole Moment (Debye)'):
                    content = f.readline().strip().split()
                    dipo = [float(content[1]), float(content[3]), float(content[5])]
                elif line.strip().startswith('Charge (ESU x 10^10)'):
                    charge = int(float(f.readline().strip()))
                elif line.strip().startswith('Polarizability Matrix (a.u.)'):
                    f.readline()
                    polarizability = [
                        list(map(float, f.readline().strip().split()[1:])),
                        list(map(float, f.readline().strip().split()[1:])),
                        list(map(float, f.readline().strip().split()[1:]))
                    ]
                    polarizability = -np.array(polarizability)
        
        coords = np.array(coords)
        dipo = np.array(dipo)
        molecule = Molecule(atoms, coords, charge=charge, dipo=dipo, polarizability=polarizability)
        return molecule
    
    @staticmethod
    def read_fda_out(out: os.PathLike, unit: Literal['kJ/mol/A', 'kcal/mol/A'] = 'kJ/mol/A'):        
        forces = {}
        natoms = None
        atoms, coords, charges = None, None, None
        _read_atoms = False
        with open(out) as f:
            for line in f:
                if line.startswith('$molecule') and (not _read_atoms):
                    atoms, coords, charges, mults, total_charge, total_mult = _parse_molecule_block_with_fragments(f)
                    natoms = sum(len(a) for a in atoms)
                    _read_atoms = True
                elif line.startswith(' Geometric Distortion Forces:'):
                    f.readline()
                    forces['geom'] = _parse_force_block(f, natoms)
                elif line.startswith(' Frozen Forces:'):
                    f.readline()
                    forces['frozen'] = _parse_force_block(f, natoms)
                elif line.startswith(' Classical Electrostatic Forces:'):
                    f.readline()
                    forces['perm_elec'] = _parse_force_block(f, natoms)
                elif line.startswith(' Non-Electrostatic Frozen Forces:'):
                    f.readline()
                    forces['pauli_disp'] = _parse_force_block(f, natoms)
                elif line.startswith(' Polarization Forces:'):
                    f.readline()
                    forces['pol'] = _parse_force_block(f, natoms)
                elif line.startswith(' Charge Transfer Forces:'):
                   f.readline()
                   forces['ct'] = _parse_force_block(f, natoms)
                elif line.startswith(' Total Forces:'):
                    f.readline()
                    forces['total'] = _parse_force_block(f, natoms)
                    break
        
        if unit == 'kcal/mol/A':
            for f in forces:
                forces[f] /= 4.184
        
        forces['interaction'] = forces['total'] - forces['geom']
        return atoms, coords, charges, forces
                

class QChemTask(Task):
    def __init__(self, wdir: os.PathLike, molecule: Molecule, config: Dict[str, Any] = dict(), name: str = '', logger: Optional[logging.Logger] = None):
        self.config = config.copy()
        self.job_type = self.config.pop('jobtype', 'opt')
        self.num_cores = self.config.pop('num_cores', 64)
        name = name if name else self.job_type
        super().__init__(name, wdir, logger)
        self.molecule = molecule
        # qchem does not accept stdin, it use input file as an argument in the command line
        self.input = self.stdin
        self.stdin = None
    
    def prep(self):
        self.cmd = f'qchem -nt {self.num_cores} {self.input}'
        writer = QChemWriter(self.input, self.job_type, self.config)
        writer.write(self.molecule)



if __name__ == '__main__':
    print(QChemReader.read_out('../monomers/water/01.opt/opt.out'))