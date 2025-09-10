import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .metrics import plot_correlation, plot_eda_scan
from ..units import HARTREE2KCAL


def report_dipos(trainer, data):
    with torch.no_grad():
        res, ref, _, _ = trainer.evaluate(data)
    dipo_cmm = np.linalg.norm(res.numpy(force=True), axis=1)
    dipo_qm = np.linalg.norm(ref.numpy(force=True), axis=1)
    # dipo_cmm = res.numpy(force=True).flatten()
    # dipo_qm = res.numpy(force=True).flatten()
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
        if 'k_index' in data.eda_df.columns:
            xdata = data.eda_df['k_index'].values
            xlabel = 'k_index' if xlabel is None else xlabel
        else:
            xdata = data.eda_df['dist'].values
            xlabel = 'dist' if xlabel is None else xlabel
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_eda_scan(
        xdata,
        ref,
        res,
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
        