import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openmm.unit as unit
from .metrics import plot_correlation, plot_eda_scan
from ..units import HARTREE2KCAL, BOHR2ANG


def report_dipos(trainer, data, report_norm=True):
    with torch.no_grad():
        res, ref, _, _ = trainer.evaluate(data)
    if report_norm:
        dipo_cmm = np.linalg.norm(res.numpy(force=True), axis=1)
        dipo_qm = np.linalg.norm(ref.numpy(force=True), axis=1)
    else:
        dipo_cmm = res.numpy(force=True).flatten()
        dipo_qm = res.numpy(force=True).flatten()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    plot_correlation(dipo_qm, dipo_cmm, 'QM Dipole Moment (a.u.)', 'CMM Dipole Moment (a.u.)', ax=ax)

def report_esp(trainer, data):
    with torch.no_grad():
        res, ref, _, _ = trainer.evaluate(data)
    esp_cmm = res.numpy(force=True)
    esp_qm = ref.numpy(force=True)
    mae = np.mean(np.abs(esp_cmm - esp_qm)) * HARTREE2KCAL
    print('MAE (kcal/mol)', mae)


def report_eda_scan(trainer, data, xdata=None, xlabel=None, **kwargs):
    with torch.no_grad():
        res, ref, _, _ = trainer.evaluate(data)
    
    if xdata is None:
        if data.eda_df is not None:
            if 'k_index' in data.eda_df.columns:
                xdata = data.eda_df['k_index'].values
                xlabel = 'k_index' if xlabel is None else xlabel
            else:
                xdata = data.eda_df['dist'].values
                xlabel = 'dist' if xlabel is None else xlabel
        else:
            masses = torch.tensor([at.element.mass.value_in_unit(unit.dalton) for at in data.top.atoms()], dtype=data.coords.dtype, device=data.coords.device).reshape(-1, 1)
            natoms = [len(list(res.atoms())) for res in data.top.residues()]
            com1 = torch.sum(data.coords[:, :natoms[0]] * masses[:natoms[0]], dim=-2) / torch.sum(masses[:natoms[0]])
            com2 = torch.sum(data.coords[:, natoms[0]:] * masses[natoms[0]:], dim=-2) / torch.sum(masses[natoms[0]:])
            com_dist = torch.norm(com1 - com2, dim=1) * BOHR2ANG
            argsort = torch.argsort(com_dist)
            for key in res:
                if torch.is_tensor(res[key]) and len(res[key]) > 0:
                    res[key] = res[key][argsort]
            for key in ref:
                if torch.is_tensor(ref[key]) and len(ref[key]) > 0:
                    ref[key] = ref[key][argsort]
            xdata = com_dist[argsort]
            xlabel = 'COM Dist. (Angstrom)' if xlabel is None else xlabel
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_eda_scan(
        xdata,
        ref,
        res,
        xlabel=xlabel,
        ax=axes[0],
        **kwargs
    )

    keys = ['perm_elec', 'pauli', 'disp', 'pol', 'ct', 'total']
    with torch.no_grad():
        error = {key: res[key] - ref[key] for key in keys}
    plot_eda_scan(
        xdata,
        error,
        xlabel=xlabel,
        ylabel='Energy Error (kcal/mol)',
        ax=axes[1],
        **kwargs
    )
    return fig


def report_eda_cluster(trainer, data, **kwargs):
    with torch.no_grad():
        res, ref, _, _ = trainer.evaluate(data)
    
    keys = ['total', 'perm_elec', 'pol', 'ct', 'pauli', 'disp']
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    axes = axes.flatten()
    for i in range(len(keys)):
        ax = axes[i]
        key = keys[i]
        plot_correlation(ref[key], res[key], xlabel='QM (kcal/mol)', ylabel='CMM (kcal/mol)', ax=ax)
        ax.set_title(key.upper())
    return fig
        