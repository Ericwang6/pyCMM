"""How the saturation parameters shape the polarization-energy scan —
in particular the quasi-linear "flat" region of the EDA polarization energy
at intermediate separations before the short-range turnover.

Four rows (Na-F, Na-Cl, Na-Br, Na-I), two columns:
  left  — sweep sat_c_iso (depth) at the fitted per-ion field scale E0
  right — sweep sat_e0 (onset field) at the fitted c_iso
EDA reference as points; the fitted parameter set drawn in black.

Usage:
    python scripts/plot_energy_sweep.py [--data /path/to/CMM_Data]
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
sys.path.insert(0, os.path.dirname(__file__))

from cmm.ffxml import ForceFieldXML
from plot_dipole_saturation import load_scan, model_pol_scan, TYPE_OF

PAIRS = ['na_f', 'na_cl', 'na_br', 'na_i']
CISO_SWEEP = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
E0_SWEEP = [0.01, 0.02, 0.04, 0.08, 0.16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(__file__), '../../CMM_Data'))
    ap.add_argument('--ff', default=os.path.join(os.path.dirname(__file__), 'ion_water_saturation.xml'))
    ap.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), 'figures'))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    root = os.path.join(args.data, 'ion_water', 'ion_ion_scans')
    ff = ForceFieldXML(args.ff, device='cpu', float_dtype=torch.float64)

    types = list(ff.pset.find('Polarization/Pol/type'))
    c_iso_t = ff.pset.find('Polarization/Pol/sat_c_iso')
    e0_t = ff.pset.find('Polarization/Pol/sat_e0')

    blues = [plt.get_cmap('Blues')(x) for x in np.linspace(0.35, 0.95, len(CISO_SWEEP))]
    oranges = [plt.get_cmap('Oranges')(x) for x in np.linspace(0.35, 0.95, len(E0_SWEEP))]

    fig, axes = plt.subplots(len(PAIRS), 2, figsize=(11, 15), constrained_layout=True)
    for row, pair in enumerate(PAIRS):
        data = load_scan(root, pair)
        r = data.eda_df['distances'].values
        e_eda = data.energies['pol'].numpy()
        idx = types.index(TYPE_OF[pair.split('_')[1]])
        c_fit, e0_fit = c_iso_t[idx].item(), e0_t[idx].item()

        for col, (sweep, ramp, label) in enumerate([
            (CISO_SWEEP, blues, '$c_{iso}$'),
            (E0_SWEEP, oranges, '$E_0$'),
        ]):
            ax = axes[row, col]
            ax.plot(r, e_eda, 'o', ms=4, color='#333333', zorder=5, label='EDA (pol)')
            try:
                for val, color in zip(sweep, ramp):
                    with torch.no_grad():
                        if col == 0:
                            c_iso_t[idx] = val
                        else:
                            e0_t[idx] = val
                    e = model_pol_scan(ff, data, use_sat=True)
                    ax.plot(r, e, '-', lw=1.6, color=color, label=f'{label} = {val:g}')
                with torch.no_grad():
                    c_iso_t[idx], e0_t[idx] = c_fit, e0_fit
                e = model_pol_scan(ff, data, use_sat=True)
                ax.plot(r, e, '-', lw=2.4, color='#1a1a1a', label='fitted', zorder=4)
            finally:
                with torch.no_grad():
                    c_iso_t[idx], e0_t[idx] = c_fit, e0_fit

            lo = e_eda.min()
            ax.set_ylim(1.8 * lo, 0.05 * abs(lo))
            fixed = (f'$E_0$={e0_fit:.3f}' if col == 0 else f'$c_{{iso}}$={c_fit:.2f}')
            ax.set_title(f'{pair.replace("_", "–")}   sweep {label}  ({fixed} fixed)', fontsize=11)
            ax.grid(alpha=0.25, lw=0.5)
            if row == len(PAIRS) - 1:
                ax.set_xlabel('ion–ion distance (Å)')
            if col == 0:
                ax.set_ylabel('E$_{pol}$ (kcal/mol)')
            if row == 0:
                ax.legend(fontsize=8, loc='lower right', ncol=2)

    fig.suptitle('Shaping the flat region of E$_{pol}$: saturation depth ($c_{iso}$) vs onset field ($E_0$)\n'
                 'sodium halides, anisotropy off', fontsize=13)
    out = os.path.join(args.outdir, 'energy_sweep_sodium_halides.png')
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
