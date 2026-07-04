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


def report_eda_scan(trainer, data, xdata=None, xlabel=None, xlim=None, **kwargs):
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
            natoms = [len(list(r.atoms())) for r in data.top.residues()]
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

    # sort by xdata
    xdata = np.asarray(xdata if not isinstance(xdata, torch.Tensor) else xdata.cpu().numpy())
    argsort = np.argsort(xdata)
    xdata = xdata[argsort]
    res = {k: v[argsort] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == len(xdata) else v for k, v in res.items()}
    ref = {k: v[argsort] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == len(xdata) else v for k, v in ref.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    plot_eda_scan(xdata, ref, res, xlabel=xlabel, ax=axes[0], **kwargs)
    axes[0].set_title('QM vs CMM')
    if xlim is not None:
        axes[0].set_xlim(xlim)

    keys = ['perm_elec', 'pauli', 'disp', 'pol', 'ct', 'total']
    with torch.no_grad():
        error = {key: res[key] - ref[key] for key in keys}
    plot_eda_scan(xdata, error, xlabel=xlabel, ylabel='Energy Error (kcal/mol)', ax=axes[1], **kwargs)
    axes[1].set_title('CMM - QM Error')
    if xlim is not None:
        axes[1].set_xlim(xlim)
    axes[1].axhline(0, color='k', lw=0.5, ls='--')

    plt.tight_layout()
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
        
def report_dimer_dipoles(trainer, data, convert_to_debye=True):
    """
    Compare CMM vs QM dipole moments for dimer (or n-mer) configurations
    loaded from EDA output files.
    
    Parameters
    ----------
    trainer : Trainer
    data    : EdaData with dimer_dipoles_qm populated
    convert_to_debye : bool, convert CMM output (a.u.) to Debye for comparison
    """
    if data.dimer_dipoles_qm is None:
        raise ValueError("No QM dipole data found. "
                         "Make sure EDA files were parsed with the updated reader.")

    AU2DEBYE = 2.541746

    # Get CMM dipoles by running getEnergy on all configs
    from cmm.topology import Topology
    system = trainer.ff.parametrize(
        Topology.fromOpenmm(data.top), batch=True
    )

    with torch.no_grad():
        res = system.getEnergy(data.coords)
        # res['dipoles'] shape: (n_configs, 3) in a.u.
        cmm_dipoles = res['dipoles']

    if convert_to_debye:
        cmm_dipoles = cmm_dipoles * AU2DEBYE

    # Compute magnitudes
    cmm_mag = torch.norm(cmm_dipoles, dim=1).numpy(force=True)
    qm_mag  = torch.norm(data.dimer_dipoles_qm, dim=1).numpy(force=True)

    # Summary statistics
    errors = cmm_mag - qm_mag
    print(f"Dipole magnitude comparison (Debye):")
    print(f"  QM  mean ± std: {qm_mag.mean():.4f} ± {qm_mag.std():.4f}")
    print(f"  CMM mean ± std: {cmm_mag.mean():.4f} ± {cmm_mag.std():.4f}")
    print(f"  Mean signed error (CMM - QM): {errors.mean():+.4f}")
    print(f"  MAE:  {np.abs(errors).mean():.4f}")
    print(f"  RMSE: {np.sqrt((errors**2).mean()):.4f}")

    # Correlation plot — magnitudes
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

    plot_correlation(
        qm_mag, cmm_mag,
        xlabel='QM |μ| (Debye)',
        ylabel='CMM |μ| (Debye)',
        ax=axes[0]
    )
    axes[0].set_title('Dipole Magnitude')

    # Component-wise: flatten x,y,z across all configs
    qm_flat  = data.dimer_dipoles_qm.numpy(force=True).flatten()
    cmm_flat = cmm_dipoles.numpy(force=True).flatten()
    plot_correlation(
        qm_flat, cmm_flat,
        xlabel='QM μ component (Debye)',
        ylabel='CMM μ component (Debye)',
        ax=axes[1]
    )
    axes[1].set_title('Dipole Components (X, Y, Z)')

    fig.suptitle('CMM vs QM Dipole Moments', fontsize=13)
    return fig

def report_polarizability(trainer):
    """
    Compare CMM vs QM molecular polarizability tensor.
    QM reference from CCSD(T) opt.out (bohr³).
    """
    # QM reference from opt.out (absolute values of negative Q-Chem output)
    QM_ALPHA = {'xx': 10.0323, 'yy': 9.4092, 'zz': 9.6595}
    QM_ISO = (QM_ALPHA['xx'] + QM_ALPHA['yy'] + QM_ALPHA['zz']) / 3

    pol_data = trainer.ff.pset.data['Polarization']['Pol']
    
    with torch.no_grad():
        alpha_xx = pol_data['alpha_xx'].detach().cpu().numpy()  # [O, H]
        alpha_yy = pol_data['alpha_yy'].detach().cpu().numpy()
        alpha_zz = pol_data['alpha_zz'].detach().cpu().numpy()
        atom_types = pol_data['type']  # ['ow', 'hw']

    # Molecular totals (O + 2H, so H contribution doubled)
    # For water: 1 O + 2 H, but pset only stores unique types
    # H appears once in pset but twice in molecule
    cmm_xx  = alpha_xx[0] + 2 * alpha_xx[1]   # O + 2H
    cmm_yy  = alpha_yy[0] + 2 * alpha_yy[1]
    cmm_zz  = alpha_zz[0] + 2 * alpha_zz[1]
    cmm_iso = (cmm_xx + cmm_yy + cmm_zz) / 3

    print("=" * 58)
    print("Molecular Polarizability Tensor (bohr³)")
    print("=" * 58)
    print(f"  {'Component':<12s}  {'CMM':>10s}  {'QM':>10s}  {'Error':>10s}")
    print(f"  {'-'*50}")
    for comp, cmm_val, qm_val in [
        ('alpha_xx', cmm_xx,  QM_ALPHA['xx']),
        ('alpha_yy', cmm_yy,  QM_ALPHA['yy']),
        ('alpha_zz', cmm_zz,  QM_ALPHA['zz']),
        ('isotropic', cmm_iso, QM_ISO),
    ]:
        err = cmm_val - qm_val
        flag = ' ✓' if abs(err) < 0.3 else ' ✗'
        print(f"  {comp:<12s}  {cmm_val:>10.4f}  {qm_val:>10.4f}  {err:>+10.4f}{flag}")
    print("=" * 58)

    # Per-type breakdown
    print(f"\nPer-type polarizabilities (bohr³):")
    print(f"  {'Type':<6s}  {'alpha_xx':>10s}  {'alpha_yy':>10s}  {'alpha_zz':>10s}  {'iso':>10s}")
    print(f"  {'-'*52}")
    for i, atype in enumerate(atom_types):
        iso_i = (alpha_xx[i] + alpha_yy[i] + alpha_zz[i]) / 3
        mult = '(×1)' if atype == 'ow' else '(×2)'
        print(f"  {atype:<6s}{mult}  {alpha_xx[i]:>10.4f}  {alpha_yy[i]:>10.4f}  {alpha_zz[i]:>10.4f}  {iso_i:>10.4f}")

    return cmm_iso, QM_ISO