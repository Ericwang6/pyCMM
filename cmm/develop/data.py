import os, glob
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, TYPE_CHECKING
import random

if TYPE_CHECKING:
    import openmm.app as app

import numpy as np
import pandas as pd
import torch

from ..units import BOHR2ANG, BOHR2NM, DEBYE2AU, SYMB2Z
from .multiwfn import MultiwfnReader
from .qchem import QChemReader


@dataclass
class EspData:
    topology: "app.Topology"
    coord: torch.Tensor
    grid: torch.Tensor
    esp: torch.Tensor
    charge: int = 0

    @classmethod
    def from_files(cls, pdb_file, multiwfn_esp_file, multiwfn_chg_file=None):
        import openmm.app as app
        
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
    topology: "app.Topology"
    coords: torch.Tensor
    dipos: torch.Tensor
    charge: int = 0
    num: int = field(init=False)

    def __post_init__(self):
        self.num = int(self.coords.shape[0])

    @classmethod
    def from_files(cls, pdb_file, qchem_out_files):
        import openmm.app as app
        
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
    topology: "app.Topology"
    coords: torch.Tensor
    pol: torch.Tensor
    charge: int = 0
    num: int = field(init=False)

    def __post_init__(self):
        self.num = int(self.coords.shape[0])
    
    @classmethod
    def from_files(cls, pdb_file, qchem_out_files):
        import openmm.app as app
        
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
    atoms: List[List[str]]
    coords: torch.Tensor
    energies: Dict[str, torch.Tensor]    
    charges: List[int] = field(default_factory=list, repr=False)
    spins: List[int] = field(default_factory=list, repr=False)
    total_charge: List[int] = None
    total_spin: List[int] = None
    num_atoms: int = field(init=False)
    num_frags: int = field(init=False)
    num: int = field(init=False)
    topology: "app.Topology" = field(default=None, kw_only=True)
    files: List[os.PathLike] = field(default_factory=list, repr=False, kw_only=True)
    eda_df: pd.DataFrame = field(default=None, kw_only=True, repr=False)

    def __post_init__(self):
        self.num = int(self.coords.shape[0])
        self.num_atoms = sum([len(a) for a in self.atoms])
        self.num_frags = len(self.atoms)
        if len(self.charges) == 0:
            self.charges = [0 for _ in range(self.num_frags)]
        if len(self.spins) == 0:
            self.spins = [1 for _ in range(self.num_frags)]
        if self.total_charge is None:
            self.total_charge = sum(self.charges)
        assert self.total_charge == sum(self.charges)
        if self.total_spin is None:
            self.total_spin = 1

    @classmethod
    def from_csv_file(cls, pdb_file, csv_file):
        import openmm.app as app
        loaded_pdb = app.PDBFile(pdb_file)
        top = loaded_pdb.topology
        atoms = [[at.element.symbol for at in residue] for residue in top.residues()]
        eda_df = pd.read_csv(csv_file)
        try:
            enes_ref = {
                "perm_elec": eda_df['CLS_ELEC'].values / 4.184,
                "pauli": eda_df['MOD_PAULI'].values / 4.184,
                'ct': eda_df['CHARGE_TRANSFER'].values / 4.184,
                'pol': eda_df['POLARIZATION'].values / 4.184,
                'disp': eda_df['DISP'].values / 4.184,
                'total': eda_df['TOTAL'].values / 4.184
            }
        except KeyError:
            # TODO: This is a band-aid. Should make the association between keys
            # used here and possible keys from the csv file user-specifiable.
            enes_ref = {
                "perm_elec": eda_df['cls_elec'].values / 4.184,
                "pauli": eda_df['mod_pauli'].values / 4.184,
                'ct': eda_df['ct'].values / 4.184,
                'pol': eda_df['pol'].values / 4.184,
                'disp': eda_df['disp'].values / 4.184,
                'total': eda_df['int'].values / 4.184
            }
        enes_ref = {key: torch.tensor(enes_ref[key]) for key in enes_ref}
        
        if 'xyz' in eda_df:
            coords = np.array([np.array(xyz.split()).reshape(-1, 3).astype(float) / BOHR2ANG for xyz in eda_df['xyz']])
        else:
            coords = np.array([loaded_pdb.getPositions(True, i_frame) for i_frame in range(loaded_pdb.getNumFrames())]) / BOHR2NM
        coords = torch.tensor(coords)

        data = cls(atoms, coords, enes_ref, topology=top, eda_df=eda_df)
        return data
    
    @classmethod
    def from_qchem_out_files(cls, pdb_file=None, out_files=list(), **kwargs):
        assert len(out_files) > 0, 'out_files must not be empty'

        if pdb_file:
            import openmm.app as app
            top = app.PDBFile(pdb_file).topology
            _atoms = [[at.element.symbol for at in residue.atoms()] for residue in top.residues()]
        else:
            top = None
            _atoms = []
        
        energies = defaultdict(list)
        coords = []
        _charges = []
        _spins = []
        _t_charge = None
        _t_spin = None
        for i, out in enumerate(out_files):
            atoms, coord, charges, spins, total_charge, total_spin, ene = QChemReader.read_eda_out(out, **kwargs)
            coord = np.vstack(coord)
            for key in ene.keys():
                energies[key].append(ene[key])
            coords.append(coord)

            if len(_atoms) > 0:
                assert _atoms == atoms, f'{atoms}, {_atoms}'
            else:
                _atoms = atoms

            if i == 0:
                _charges = charges
                _spins = spins
                _t_charge = total_charge
                _t_spin = total_spin
            else:
                assert charges == _charges
                assert spins == _spins
                assert total_charge == _t_charge
                assert total_spin == _t_spin

        coords = torch.tensor(np.array(coords) / BOHR2ANG)

        energies = {
            "perm_elec": torch.tensor(energies['CLS_ELEC']),
            "pauli": torch.tensor(energies['MOD_PAULI']),
            'ct': torch.tensor(energies['CHARGE_TRANSFER']),
            'pol': torch.tensor(energies['POLARIZATION']),
            'disp': torch.tensor(energies['DISP']),
            'total': torch.tensor(energies['TOTAL'])
        }
        
        data = cls(
            _atoms, coords, energies, topology=top, 
            charges=_charges, spins=_spins, total_charge=_t_charge, total_spin=_t_spin, 
            files=out_files
        )
        return data
    
    @classmethod
    def from_files(cls, pdb_file, file, **kwargs):
        if isinstance(file, str):
            return cls.from_csv_file(pdb_file, file)
        else:
            return cls.from_qchem_out_files(pdb_file, file, **kwargs)
    
    @staticmethod
    def gather_dimer_scan_eda(dirpath, smi0="", smi1=""):
        df = []
        for edaout in glob.glob(f'{dirpath}/k_*/eda.out'):
            atoms, coords, charges, spins, total_charge, total_spin, res = QChemReader.read_eda_out(edaout, return_coords=True)
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
        
        if self.files is not None:
            if torch.is_tensor(idx):
                idx = idx.numpy(force=True)
            sliced_files = np.array(self.files)[idx].tolist()
        else:
            sliced_files = None

        # Create a new instance of EdaData with the sliced data
        return EdaData(
            self.atoms, sliced_coords, sliced_energies, 
            charges=self.charges, spins=self.spins,
            total_charge=self.total_charge, total_spin=self.total_spin,
            topology=self.topology, eda_df=sliced_eda_df, files=sliced_files
        )
    
    def get_ase_atoms(self):
        from ase import Atoms

        natoms_acc = [0]
        for i in range(self.num_frags):
            natoms_acc.append(natoms_acc[-1]+len(self.atoms[i]))
        
        symbols_total = []
        for a in self.atoms:
            symbols_total += a
         
        coords_numpy = self.coords.numpy(force=True) * BOHR2ANG
        frags_list = []
        for n in range(self.num):
            frags = []
            for f in range(self.num_frags):
                ats = Atoms(symbols=self.atoms[f], positions=coords_numpy[n][natoms_acc[f]:natoms_acc[f+1]])
                ats.info.update({"spin": self.spins[f], "charge": self.charges[f]})
                frags.append(ats)
            total = Atoms(symbols=symbols_total, positions=coords_numpy[n])
            total.info.update({"spin": self.total_spin, "charge": self.total_charge})
            frags_list.append((total, frags))
        return frags_list


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
