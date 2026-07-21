import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def as_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.numpy(force=True)
    else:
        return np.array(arr)


def report_metrics(y_pred, y_true):
    mean_signed_error = np.mean(y_pred - y_true)
    mean_unsigned_error = np.mean(np.abs(y_pred - y_true))
    mean_squared_error = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mean_squared_error)

    metrics = {
        "y_true": y_true,
        "y_pred": y_pred,
        "mean_signed_error": mean_signed_error,
        "mue": mean_unsigned_error,
        "mean_squared_error": mean_squared_error,
        "rmse": rmse
    }
    return metrics


def plot_correlation(xdata, ydata, xlabel, ylabel, ax=None, calc_mae=True, calc_mse=True):
    xdata = as_numpy(xdata)
    ydata = as_numpy(ydata)

    if calc_mae:
        mae = np.mean(np.abs(xdata - ydata))
        label = f'MAE: {mae:.4f}'
    else:
        label = None
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    if len(xdata) > 1:
        data = np.vstack((xdata, ydata))
        kde = gaussian_kde(data)
        density = kde(data)
    else:
        density = None

    label_mse = f"MSE: {np.mean(ydata - xdata):.4f}" if calc_mse else None
    ax.scatter(xdata, ydata, label=label_mse, s=10, c=density, cmap='plasma')
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    vmin, vmax = min([xmin, ymin]), max([xmax, ymax])
    ax.plot([vmin, vmax], [vmin, vmax], linestyle='--', color='black', label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xticks, yticks = ax.get_xticks(), ax.get_yticks()
    ticks = xticks if len(xticks) > len(yticks) else yticks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.legend()
    ax.grid(True)
    return ax 


EDA_COLORS = {
    "perm_elec": "blue",
    "pauli": "red",
    "pol": "green",
    "ct": "orange",
    "disp": "purple",
    "total": "black"
}
    

# def plot_eda_scan(xdata, ene_ref, ene_cmm=None, ax=None, keys=list(), xlabel='k_index', ylabel='Energy (kcal/mol)', xmin=-np.inf, xmax=np.inf, ymin=None, ymax=None):
    
#     if ymin is not None:
#         plt.ylim(bottom=ymin)
#     if ymax is not None:
#         plt.ylim(top=ymax)

#     if ax is None:
#         fig, ax = plt.subplots(1, 1, constrained_layout=True)
    

#     xdata = as_numpy(xdata)
#     mask = np.logical_and(xdata < xmax, xdata > xmin)

#     xdata = xdata[mask]
#     ene_ref = {key: as_numpy(ene_ref[key])[mask] for key in ene_ref}
#     if ene_cmm is not None:
#         ene_cmm = {key: as_numpy(ene_cmm[key])[mask] for key in ene_ref}
#     # Add this
#     print("ene_cmm is None?", ene_cmm is None)
#     print("ene_cmm keys:", list(ene_cmm.keys()) if ene_cmm is not None else "N/A")
#     for key in keys:
#         print(f"key={key}, in ene_ref={key in ene_ref}, in ene_cmm={ene_cmm is not None and key in ene_cmm}")
    
#     if len(keys) == 0:
#         keys = list(EDA_COLORS.keys())
    
#     metrics = {}
#     for key in keys:
#         ax.plot(xdata, ene_ref[key], 'o-', color=EDA_COLORS[key], label=key)
#         if ene_cmm:
#             ax.plot(xdata, ene_cmm[key], 'o--', color=EDA_COLORS[key])
#             metrics[key] = report_metrics(ene_cmm[key], ene_ref[key])
    
#     ax.legend()
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     return metrics
def plot_eda_scan(xdata, ene_ref, ene_cmm=None, ax=None, keys=list(), xlabel='k_index', ylabel='Energy (kcal/mol)', xmin=-np.inf, xmax=np.inf, ymin=None, ymax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    xdata = as_numpy(xdata)
    mask = np.logical_and(xdata < xmax, xdata > xmin)
    xdata = xdata[mask]

    ene_ref = {key: as_numpy(ene_ref[key])[mask] for key in ene_ref}
    if ene_cmm is not None:
        ene_cmm = {key: as_numpy(ene_cmm[key])[mask] for key in ene_ref if key in ene_cmm}

    if len(keys) == 0:
        keys = list(EDA_COLORS.keys())

    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)

    metrics = {}
    for key in keys:
        if key not in ene_ref:
            continue
        ax.plot(xdata, ene_ref[key], 'o-', color=EDA_COLORS[key], label=key)
        if ene_cmm is not None and key in ene_cmm:
            ax.plot(xdata, ene_cmm[key], 'o--', color=EDA_COLORS[key])
            metrics[key] = report_metrics(ene_cmm[key], ene_ref[key])

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return metrics