import sys, os
import glob
import json
import random
from typing import List, Dict
from dataclasses import dataclass, field
from pathlib import Path

import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openmm.app as app

from cmm.forcefield import CMMForceField
from cmm.units import HARTREE2KCAL, BOHR2ANG, BOHR2NM
from cmm.develop.optimize import Trainer2, Optimizer
from cmm.develop.metrics import plot_correlation, plot_eda_scan
from cmm.develop.extract_params import extract_bond_angle_eq, extract_multipoles
from cmm.misc_utils import *

def report_eda_cluster(trainer, data, **kwargs):
    with torch.no_grad():
        res, ref = trainer.evaluate(data)

    keys = ['total', 'perm_elec', 'pol', 'ct', 'pauli', 'disp']
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    axes = axes.flatten()
    for i in range(len(keys)):
        ax = axes[i]
        key = keys[i]
        if key == 'total':
            plot_correlation(ref['int'], res[key], xlabel='QM (kcal/mol)', ylabel='CMM (kcal/mol)', ax=ax)
        else:
            plot_correlation(ref[key], res[key], xlabel='QM (kcal/mol)', ylabel='CMM (kcal/mol)', ax=ax)
        ax.set_title(key.upper())
    return fig

@dataclass
class EdaData:
    topologies: List[app.Topology]
    coords: List[torch.Tensor]
    energies: Dict[str, torch.Tensor]
    eda_df: pd.DataFrame = None
    num: int = field(init=False)

    def __post_init__(self):
        self.num = int(len(self.coords))

    @classmethod
    def from_csv_file(cls, pdb_file, csv_file, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        topologies, coords = cls.get_openmm_topologies_and_coordinates(pdb_file)
        coords = [torch.from_numpy(xyz).to(device) for xyz in coords]

        eda_df = pd.read_csv(csv_file)
        enes_ref = {
            'perm_elec': eda_df['cls_elec'].values / 4.184,
            'pauli': eda_df['mod_pauli'].values / 4.184,
            'ct': eda_df['ct'].values / 4.184,
            'pol': eda_df['pol'].values / 4.184,
            'disp': eda_df['disp'].values / 4.184,
            'int': eda_df['int'].values / 4.184
        }
        enes_ref = {key: torch.tensor(enes_ref[key]).to(device) for key in enes_ref}

        data = cls(topologies, coords, enes_ref, eda_df)
        return data

    @staticmethod
    def get_openmm_topologies_and_coordinates(pdb_file: str):
        pdb = app.PDBFile(pdb_file)
        n_frames = pdb.getNumFrames()

        topologies = []
        coords = []

        for frame_index in range(n_frames):
            frame_content = extract_frame_as_pdb_string(pdb_file, frame_index)
            with temporary_pdb_file(frame_content) as temp_pdb_path:
                topology = app.PDBFile(temp_pdb_path).topology
                topologies.append(topology)
                coords.append(pdb.getPositions(True, frame_index)._value / BOHR2NM)
        return topologies, coords

    def __getitem__(self, idx):
        sliced_coords = self.coords[idx]
        sliced_energies = {key: value[idx] for key, value in self.energies.items()}
        if self.eda_df is not None:
            if isinstance(idx, int):
                sliced_eda_df = self.eda_df.iloc[[idx]]
            else:
                sliced_eda_df = self.eda_df.iloc[idx]
        else:
            sliced_eda_df = None
        return EdaData(self.topologies[idx], sliced_coords, sliced_energies, sliced_eda_df)

home = Path.home()
data_path = os.path.join(home, "dev/CMM_Data/ion_water")

# TODO: The specific data used for training should actually be selected by
# a command-line argument from some configuration file.

### Set up data to be used for training ###
f_water_pdb = os.path.join(data_path, "train_test_splits/f_water_wb97xv_qzvppd_train.pdb")
f_water_csv = os.path.join(data_path, "train_test_splits/f_water_wb97xv_qzvppd_train.csv")
data = EdaData.from_csv_file(f_water_pdb, f_water_csv)

print(f"Current number of threads: {torch.get_num_threads()}")
print(f"Inter-op threads: {torch.get_num_interop_threads()}")

### Set up force field, optimizer, and trainer for evaluation ###
ff = CMMForceField(os.path.join(home, "dev/pyCMM/tests/data/ion_refit.json"))

optimizer = Optimizer(
    ff,
    freeze_water=True,
    opt_params={
        "atomic_params": [
            'Z', 'b_elec',
            'b_pauli', 'q_pauli',
            'q_xpol', 'b_xpol',
            'C6_disp', 'b_disp',
            'b_ct', 'q_ct_acc', 'q_ct_don'
        ],
        "pair_params": ['eps_ct'],
    },
    optim='adam',
    lr=0.05,
    # freeze_types={"atomic_params": ['hw']}
    # l2=1.0,
    # l2_params={'atomic_params': ['Z', 'b_elec']}
)

trainer = Trainer2(
    ff,
    optimizer,
    target_weights={EdaData:1},
    eda_weights={'perm_elec': 1.0, 'total': 0.0, 'pauli': 1.0, 'disp': 1.0, 'pol': 1.0, 'ct': 1.0, 'int': 0.0}
)

### ALL ABOARD ###
n_macro = 20
n_epochs = 50
for i_macro in range(n_macro):
    trainer.train(data, num_epoch=n_epochs)

    ### Record Parameters and Visualize Losses ###
    ff.save(f'fit_params_{(i_macro+1)*n_epochs}.json')
    fig = report_eda_cluster(trainer, data)
    fig.savefig(f'train_correlation{(i_macro+1)*n_epochs}.png')