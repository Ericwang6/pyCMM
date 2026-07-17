"""Four-panel plot: how the finite-field pair polarizability of the sodium
halides varies with the isotropic saturation coupling sat_c_iso.

For each Na-X scan the halide's sat_c_iso is swept over a fixed range while
its fitted per-ion field scale sat_e0 (and sat_c_ani = 0) is held at the
fitted value. Parallel component solid, perpendicular dashed; the fitted
c_iso is drawn on top in black.

Usage:
    python scripts/plot_ciso_sweep.py [--data /path/to/CMM_Data]
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
from plot_dipole_saturation import (
    load_scan, free_ion_alpha, pair_polarizability_scan, TYPE_OF
)

PAIRS = ['na_f', 'na_cl', 'na_br', 'na_i']
SWEEP = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]


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
    c_iso_tensor = ff.pset.find('Polarization/Pol/sat_c_iso')
    e0_tensor = ff.pset.find('Polarization/Pol/sat_e0')

    # sequential ramp: light -> dark = small -> large c_iso (magnitude)
    cmap = plt.get_cmap('Blues')
    ramp = [cmap(x) for x in np.linspace(0.35, 0.95, len(SWEEP))]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for panel, pair in enumerate(PAIRS):
        ax = axes.flat[panel]
        data = load_scan(root, pair)
        r = data.eda_df['distances'].values
        cat, an = pair.split('_')
        idx = types.index(TYPE_OF[an])
        c_fit = c_iso_tensor[idx].item()
        e0_fit = e0_tensor[idx].item()
        a_free = free_ion_alpha(ff, TYPE_OF[cat]) + free_ion_alpha(ff, TYPE_OF[an])

        try:
            for c, color in zip(SWEEP, ramp):
                with torch.no_grad():
                    c_iso_tensor[idx] = c
                par, perp = pair_polarizability_scan(ff, data, use_sat=True)
                ax.plot(r, par / a_free, '-', lw=1.6, color=color)
                ax.plot(r, perp / a_free, '--', lw=1.3, color=color)
            # fitted value on top
            with torch.no_grad():
                c_iso_tensor[idx] = c_fit
            par, perp = pair_polarizability_scan(ff, data, use_sat=True)
            ax.plot(r, par / a_free, '-', lw=2.4, color='#1a1a1a')
            ax.plot(r, perp / a_free, '--', lw=2.0, color='#1a1a1a')
        finally:
            with torch.no_grad():
                c_iso_tensor[idx] = c_fit

        ax.axhline(1.0, color='#999999', lw=1, ls=':')
        ax.set_title(f'{cat.capitalize()}–{an.capitalize()}   '
                     f'(fit: $c_{{iso}}$={c_fit:.2f}, $E_0$={e0_fit:.3f} a.u.)',
                     fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
        if panel >= 2:
            ax.set_xlabel('ion–ion distance (Å)')
        if panel % 2 == 0:
            ax.set_ylabel(r'$\alpha_{pair}\;/\;\sum \alpha_{free\ ion}$')

    handles = [plt.Line2D([], [], color=c, lw=2, label=f'$c_{{iso}}$ = {v:g}')
               for v, c in zip(SWEEP, ramp)]
    handles.append(plt.Line2D([], [], color='#1a1a1a', lw=2.4, label='fitted $c_{iso}$'))
    handles += [
        plt.Line2D([], [], color='#666666', ls='-', lw=2, label=r'$\alpha_\parallel$'),
        plt.Line2D([], [], color='#666666', ls='--', lw=1.6, label=r'$\alpha_\perp$'),
    ]
    fig.legend(handles=handles, loc='outside right center', fontsize=10, frameon=False)
    fig.suptitle('Sodium halides: pair polarizability vs isotropic saturation strength\n'
                 '(per-ion fitted field scale $E_0$ held fixed, $c_{ani}=0$)', fontsize=13)
    out = os.path.join(args.outdir, 'ciso_sweep_sodium_halides.png')
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
