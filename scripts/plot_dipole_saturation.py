"""Plots for the dipole-saturation model.

1. figures/scan_energies.png — model vs EDA polarization energy along every
   +/- ion-ion scan (fitted saturation model vs the no-saturation baseline).
2. figures/pair_polarizability_<set>.png — finite-field pair polarizability
   ratio alpha / sum(alpha_free_ion) along the scans, parallel (solid) and
   perpendicular (dashed), in the style of Fig. 1a of the CMM ion paper,
   with and without saturation.

Usage:
    python scripts/plot_dipole_saturation.py [--data /path/to/CMM_Data]
        [--ff scripts/ion_water_saturation.xml] [--outdir scripts/figures]
"""
import argparse
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.develop.data import EdaData

CATIONS = ['li', 'na', 'k', 'rb', 'cs', 'mg', 'ca']
ANIONS = ['f', 'cl', 'br', 'i']

# Okabe-Ito subset, validated (CVD floor band covered by direct labels)
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
C_MODEL = '#0072B2'   # saturation model
C_BASE = '#D55E00'    # no-saturation baseline
C_REF = '#333333'     # EDA reference

TYPE_OF = {'li': 'li+', 'na': 'na+', 'k': 'k+', 'rb': 'rb+', 'cs': 'cs+',
           'mg': 'mg2+', 'ca': 'ca2+', 'f': 'f-', 'cl': 'cl-', 'br': 'br-', 'i': 'i-'}


def load_scan(root, pair):
    data = EdaData.from_csv_file(os.path.join(root, f'{pair}_scan.pdb'),
                                 os.path.join(root, f'{pair}_scan.csv'))
    data.name = pair
    return data


def free_ion_alpha(ff, atom_type):
    types = list(ff.pset.find('Polarization/Pol/type'))
    i = types.index(atom_type)
    axx = ff.pset.find('Polarization/Pol/alpha_xx')[i].item()
    ayy = ff.pset.find('Polarization/Pol/alpha_yy')[i].item()
    azz = ff.pset.find('Polarization/Pol/alpha_zz')[i].item()
    return (axx + ayy + azz) / 3.0


def model_pol_scan(ff, data, use_sat=True):
    system = ff.parametrize(Topology.fromOpenmm(data.top), batch=True,
                            use_dipole_saturation=use_sat)
    res = system.getEnergy(data.coords, energy_in_kcal=True, include_bonded=False)
    return res['pol'].detach().numpy()


def finite_field_polarizability(system, coords, h=1e-4):
    """Batched finite-field molecular polarizability, shape (nbz, 3, 3)."""
    nbz = coords.shape[0]
    a = torch.zeros(nbz, 3, 3)
    for b in range(3):
        E = torch.zeros(3)
        E[b] = h
        Pp = system.getEnergy(coords, ext_field=E)['induced_molecular_dipole'].detach()
        Pm = system.getEnergy(coords, ext_field=-E)['induced_molecular_dipole'].detach()
        a[:, :, b] = (Pp - Pm) / (2 * h)
    return 0.5 * (a + a.transpose(1, 2))


def pair_polarizability_scan(ff, data, use_sat=True):
    """Parallel/perpendicular pair polarizability along a 2-ion scan."""
    system = ff.parametrize(Topology.fromOpenmm(data.top), batch=True,
                            use_dipole_saturation=use_sat)
    alphas = finite_field_polarizability(system, data.coords)
    axis = data.coords[:, 1] - data.coords[:, 0]
    axis = axis / torch.norm(axis, dim=1, keepdim=True)
    par = torch.einsum('ni,nij,nj->n', axis, alphas, axis)
    perp = 0.5 * (torch.einsum('nii->n', alphas) - par)
    return par.numpy(), perp.numpy()


def plot_scan_energies(ff, root, outpath):
    quad_pol_t = ff.pset.find('Polarization/Pol/quad_pol')
    has_quad = bool(torch.any(quad_pol_t != 0))
    fig, axes = plt.subplots(len(CATIONS), len(ANIONS), figsize=(15, 21),
                             sharex=False, constrained_layout=True)
    for i, cat in enumerate(CATIONS):
        for j, an in enumerate(ANIONS):
            ax = axes[i, j]
            pair = f'{cat}_{an}'
            data = load_scan(root, pair)
            r = data.eda_df['distances'].values
            e_eda = data.energies['pol'].numpy()
            e_sat = model_pol_scan(ff, data, use_sat=True)
            e_base = model_pol_scan(ff, data, use_sat=False)

            ax.plot(r, e_eda, 'o', ms=4, color=C_REF, label='EDA (pol)', zorder=3)
            ax.plot(r, e_base, '--', lw=2, color=C_BASE, label='CMM, no saturation')
            if has_quad:
                quad_backup = quad_pol_t.detach().clone()
                try:
                    with torch.no_grad():
                        quad_pol_t.zero_()
                    e_sat_only = model_pol_scan(ff, data, use_sat=True)
                finally:
                    with torch.no_grad():
                        quad_pol_t.copy_(quad_backup)
                ax.plot(r, e_sat_only, ':', lw=2, color='#009E73', label='CMM, saturation only')
                ax.plot(r, e_sat, '-', lw=2, color=C_MODEL, label='CMM, saturation + quad pol')
            else:
                ax.plot(r, e_sat, '-', lw=2, color=C_MODEL, label='CMM, saturation')

            lo = min(e_eda.min(), e_sat.min())
            ax.set_ylim(1.3 * lo, 2.0)
            ax.set_title(pair.replace('_', '–'), fontsize=11)
            ax.grid(alpha=0.25, lw=0.5)
            if i == len(CATIONS) - 1:
                ax.set_xlabel('ion–ion distance (Å)')
            if j == 0:
                ax.set_ylabel('E$_{pol}$ (kcal/mol)')
    axes[0, 0].legend(fontsize=9, loc='lower right')
    fig.suptitle('Polarization energy along +/- ion–ion scans', fontsize=14)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'wrote {outpath}')


def plot_polarizability_ratio(ff, root, pairs, labels, outpath, title):
    """Fig-1a-style: alpha_par (solid) and alpha_perp (dashed) over the free-ion
    sum, with saturation (right) and without (left)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True,
                             constrained_layout=True)
    for k, (pair, label) in enumerate(zip(pairs, labels)):
        data = load_scan(root, pair)
        r = data.eda_df['distances'].values
        cat, an = pair.split('_')
        a_free = free_ion_alpha(ff, TYPE_OF[cat]) + free_ion_alpha(ff, TYPE_OF[an])
        color = COLORS[k % len(COLORS)]
        for ax, use_sat in zip(axes, (False, True)):
            par, perp = pair_polarizability_scan(ff, data, use_sat=use_sat)
            ax.plot(r, par / a_free, '-', lw=2, color=color)
            ax.plot(r, perp / a_free, '--', lw=1.6, color=color)
            if use_sat:
                ax.annotate(label, (r[-1], (par / a_free)[-1]), xytext=(4, 0),
                            textcoords='offset points', fontsize=9, color=color,
                            va='center')
    for ax, name in zip(axes, ('no saturation', 'with saturation')):
        ax.axhline(1.0, color='#999999', lw=1, ls=':')
        ax.set_xlabel('ion–ion distance (Å)')
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r'$\alpha_{pair}\;/\;\sum \alpha_{free\ ion}$')
    # line-style legend (identity is per-pair color + direct label)
    style_handles = [
        plt.Line2D([], [], color='#333333', ls='-', lw=2, label=r'$\alpha_\parallel$'),
        plt.Line2D([], [], color='#333333', ls='--', lw=1.6, label=r'$\alpha_\perp$'),
    ]
    axes[1].legend(handles=style_handles, fontsize=10, loc='upper right')
    fig.suptitle(title, fontsize=13)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'wrote {outpath}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(__file__), '../../CMM_Data'))
    ap.add_argument('--ff', default=os.path.join(os.path.dirname(__file__), 'ion_water_saturation.xml'))
    ap.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), 'figures'))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    root = os.path.join(args.data, 'ion_water', 'ion_ion_scans')
    ff = ForceFieldXML(args.ff, device='cpu', float_dtype=torch.float64)

    plot_scan_energies(ff, root, os.path.join(args.outdir, 'scan_energies.png'))

    plot_polarizability_ratio(
        ff, root,
        pairs=['li_cl', 'na_cl', 'k_cl', 'rb_cl', 'cs_cl'],
        labels=['Li–Cl', 'Na–Cl', 'K–Cl', 'Rb–Cl', 'Cs–Cl'],
        outpath=os.path.join(args.outdir, 'pair_polarizability_chlorides.png'),
        title='Pair polarizability ratio: alkali chlorides (finite field)'
    )
    plot_polarizability_ratio(
        ff, root,
        pairs=['na_f', 'na_cl', 'na_br', 'na_i'],
        labels=['Na–F', 'Na–Cl', 'Na–Br', 'Na–I'],
        outpath=os.path.join(args.outdir, 'pair_polarizability_sodium_halides.png'),
        title='Pair polarizability ratio: sodium halides (finite field)'
    )


if __name__ == '__main__':
    main()
