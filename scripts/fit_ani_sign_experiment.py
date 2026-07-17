"""Experiment: what does the anisotropic saturation term do when sat_c_ani is
allowed to go negative?

Warm-starts from the fitted isotropic model (ion_water_saturation.xml) and
re-runs the anisotropic stages with the sign unconstrained. Writes the result
to ion_water_saturation_negani.xml (the main fitted xml is left untouched).
"""
import os
import sys

import torch

torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from cmm.ffxml import ForceFieldXML
from fit_dipole_saturation import load_scans, run_stage, report, halide_mask, PAIRS

HERE = os.path.dirname(__file__)


def main():
    root = os.path.join('/Users/joseph.heindel/dev/CMM_Data', 'ion_water', 'ion_ion_scans')
    datas = load_scans(root, PAIRS)

    ff = ForceFieldXML(os.path.join(HERE, 'ion_water_saturation.xml'),
                       device='cpu', float_dtype=torch.float64, requires_grad=True)
    mask = halide_mask(ff).bool()
    with torch.no_grad():
        ff.pset.find('Polarization/Pol/sat_c_ani').data[mask] = 0.1
        ff.pset.find('Polarization/Pol/sat_w').data[mask] = 1.0

    print('===== baseline (fitted isotropic model) =====')
    with torch.no_grad():
        ff.pset.find('Polarization/Pol/sat_c_ani').data[mask] = 0.0
    report(ff, datas)
    with torch.no_grad():
        ff.pset.find('Polarization/Pol/sat_c_ani').data[mask] = 0.1

    print('\n===== stage 3 (sign-free): sat_c_ani + sat_w, isotropic frozen =====')
    run_stage(ff, datas, [('Polarization/Pol/sat_c_ani', 0.005),
                          ('Polarization/Pol/sat_w', 0.02)], 150)
    report(ff, datas)

    print('\n===== stage 4 (sign-free): joint fine-tune =====')
    run_stage(ff, datas, [('Polarization/Pol/sat_c_iso', 0.01),
                          ('Polarization/Pol/sat_e0', 0.001),
                          ('Polarization/Pol/sat_c_ani', 0.002),
                          ('Polarization/Pol/sat_w', 0.01)], 150)
    report(ff, datas)

    types = ff.pset.find('Polarization/Pol/type')
    c_iso = ff.pset.find('Polarization/Pol/sat_c_iso')
    e0 = ff.pset.find('Polarization/Pol/sat_e0')
    c_ani = ff.pset.find('Polarization/Pol/sat_c_ani')
    w = ff.pset.find('Polarization/Pol/sat_w')
    print('\n=== fitted parameters (sign-free c_ani) ===')
    for i, t in enumerate(types):
        print(f'{t:6s} sat_c_iso={c_iso[i].item():10.5f} sat_e0={e0[i].item():10.5f} '
              f'sat_c_ani={c_ani[i].item():10.5f} sat_w={w[i].item():10.5f}')

    out = os.path.join(HERE, 'ion_water_saturation_negani.xml')
    ff.save(out)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
