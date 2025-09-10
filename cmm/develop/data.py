import os, glob
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Iterable
import random

import openmm.app as app
import numpy as np
import pandas as pd
import torch

from ..units import BOHR2ANG, BOHR2NM, DEBYE2AU
from .multiwfn import MultiwfnReader
from .qchem import QChemReader


@dataclass
class EspData:
    top: app.Topology
    coord: torch.Tensor
    grid: torch.Tensor
    esp: torch.Tensor
    charge: int = 0

    @classmethod
    def from_files(cls, pdb_file, multiwfn_esp_file, multiwfn_chg_file=None):
        pdb = app.PDBFile(pdb_file)
        if multiwfn_chg_file:
            coord = MultiwfnReader.read_chg_file(multiwfn_chg_file)[1]
            coord /= BOHR2ANG
        else:
            coord = np.array([[v.x, v.y, v.z] for v in pdb.positions._value])
            coord /= BOHR2NM
        
        coord = torch.tensor(coord)

        grid, esp = MultiwfnReader.read_esp_file(multiwfn_esp_file)
        grid = torch.tensor(grid)
        esp = torch.tensor(esp)

        return cls(pdb.topology, coord, grid, esp)


@dataclass
class DipoleData:
    top: app.Topology
    coords: torch.Tensor
    dipos: torch.Tensor
    charge: int = 0
    num: int = field(init=False)

    def __post_init__(self):
        self.num = int(self.coords.shape[0])

    @classmethod
    def from_files(cls, pdb_file, qchem_out_files):
        top = app.PDBFile(pdb_file).topology
        coords, dipos = [], []
        for file in qchem_out_files:
            mol = QChemReader.read_out(file)
            coords.append(mol.coords)
            dipos.append(mol.dipo)
        coords = torch.tensor(np.array(coords) / BOHR2ANG)
        dipos = torch.tensor(np.array(dipos) * DEBYE2AU)
        return cls(top, coords, dipos)


@dataclass
class PolarizabilityData:
    top: app.Topology
    coords: torch.Tensor
    pol: torch.Tensor
    charge: int = 0
    num: int = field(init=False)

    def __post_init__(self):
        self.num = int(self.coords.shape[0])
    
    @classmethod
    def from_files(cls, pdb_file, qchem_out_files):
        top = app.PDBFile(pdb_file).topology
        coords, pols = [], []
        for file in qchem_out_files:
            mol = QChemReader.read_out(file)
            coords.append(mol.coords)
            pols.append(mol.polarizability)
        coords = torch.tensor(np.array(coords) / BOHR2ANG)
        pols = torch.tensor(np.array(pols))
        return cls(top, coords, pol=pols, charge=mol.charge)
    

@dataclass
class EdaData:
    top: app.Topology
    coords: torch.Tensor
    energies: Dict[str, torch.Tensor]
    eda_df: pd.DataFrame = None
    num: int = field(init=False)

    def __post_init__(self):
        self.num = int(self.coords.shape[0])

    @classmethod
    def from_csv_file(cls, pdb_file, csv_file):
        top = app.PDBFile(pdb_file).topology
        eda_df = pd.read_csv(csv_file)
        enes_ref = {
            "perm_elec": eda_df['CLS_ELEC'].values / 4.184,
            "pauli": eda_df['MOD_PAULI'].values / 4.184,
            'ct': eda_df['CHARGE_TRANSFER'].values / 4.184,
            'pol': eda_df['POLARIZATION'].values / 4.184,
            'disp': eda_df['DISP'].values / 4.184,
            'total': eda_df['TOTAL'].values / 4.184
        }
        enes_ref = {key: torch.tensor(enes_ref[key]) for key in enes_ref}
        coords = np.array([np.array(xyz.split()).reshape(-1, 3).astype(float) / BOHR2ANG for xyz in eda_df['xyz']])
        coords = torch.tensor(coords)

        data = cls(top, coords, enes_ref, eda_df)
        return data
    
    @classmethod
    def from_qchem_out_files(cls, pdb_file, out_files):
        top = app.PDBFile(pdb_file).topology
        coords = []
        energies = defaultdict(list)
        for out in out_files:
            _, coord, _, ene = QChemReader.read_eda_out(out)
            coord = np.vstack(coord)
            for key in ene.keys():
                energies[key].append(ene[key])
            coords.append(coord)

        coords = torch.tensor(np.array(coords) / BOHR2ANG)

        energies = {
            "perm_elec": torch.tensor(energies['CLS_ELEC']),
            "pauli": torch.tensor(energies['MOD_PAULI']),
            'ct': torch.tensor(energies['CHARGE_TRANSFER']),
            'pol': torch.tensor(energies['POLARIZATION']),
            'disp': torch.tensor(energies['DISP']),
            'total': torch.tensor(energies['TOTAL'])
        }
        
        data = cls(top, coords, energies)
        return data
    
    @classmethod
    def from_files(cls, pdb_file, file):
        if isinstance(file, str):
            return cls.from_csv_file(pdb_file, file)
        else:
            return cls.from_qchem_out_files(pdb_file, file)
    

    @staticmethod
    def gather_dimer_scan_eda(dirpath, smi0="", smi1=""):
        df = []
        for edaout in glob.glob(f'{dirpath}/k_*/eda.out'):
            atoms, coords, charges, res = QChemReader.read_eda_out(edaout, return_coords=True)
            res['smiles0'] = smi0
            res['smiles1'] = smi1
            res['k_index'] = int(os.path.basename(os.path.dirname(edaout)).split('_')[-1])
            res['elements'] = ' '.join(' '.join(atoms[i]) for i in range(len(atoms)))
            res['xyz'] = ' '.join(' '.join(f'{c:.4f}' for c in coords[i].flatten()) for i in range(len(coords)))

            for i in range(len(atoms)):
                res[f'charge{i}'] = charges[i]
                res[f'natoms{i}'] = len(atoms[i])

            df.append(res)

        df = pd.DataFrame(df).sort_values('k_index')
        df.to_csv(f'{dirpath}/eda.csv', index=None)
    
    def __getitem__(self, idx):
        """
        Enables slicing of EdaData objects. The slicing is applied to the coordinates,
        the energies (each tensor), and the eda_df (if available).

        Parameters
        ----------
        idx : int, slice, list, or any valid index for torch.Tensor and pd.DataFrame
            The indices to select.

        Returns
        -------
        EdaData
            A new EdaData instance with the sliced data.
        """
        # Slice the coordinates
        sliced_coords = self.coords[idx]

        # Slice the energies dictionary for each energy component
        sliced_energies = {key: value[idx] for key, value in self.energies.items()}

        # If eda_df is available, slice it. Use .iloc for pandas slicing.
        if self.eda_df is not None:
            # If idx is an integer, wrap it in a list to keep the result as a DataFrame
            if isinstance(idx, int):
                sliced_eda_df = self.eda_df.iloc[[idx]]
            else:
                sliced_eda_df = self.eda_df.iloc[idx]
        else:
            sliced_eda_df = None

        # Create a new instance of EdaData with the sliced data
        return EdaData(self.top, sliced_coords, sliced_energies, sliced_eda_df)


def slice_data(data: EdaData, num: int):
    chunks = []
    chunk_size = int(data.num / num)
    for i in range(num):
        chunks.append(chunk_size)
    for i in range(data.num % num):
        chunks[i] += 1
    chunks.insert(0, 0)
    chunks = np.cumsum(chunks)
    return [data[chunks[i]: chunks[i+1]] for i in range(num)]
