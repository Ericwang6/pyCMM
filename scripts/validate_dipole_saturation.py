"""Held-out validation of the fitted dipole-saturation parameters.

None of these scans entered the fit:
- like-charge ion-ion scans (anion-anion, cation-cation)
- ion-water scans (the saturation must not degrade them; water and the
  cations have sat_* = 0, so only the halide term can act)

Reports the pol-channel RMSE with and without saturation.

Usage:
    python scripts/validate_dipole_saturation.py [--data /path/to/CMM_Data]
        [--ff scripts/ion_water_saturation.xml]
"""
import argparse
import os
import sys

import torch

torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cmm.ffxml import ForceFieldXML
from cmm.topology import Topology
from cmm.develop.data import EdaData

LIKE_CHARGE = ['f_f', 'cl_cl', 'br_br', 'i_i', 'li_li', 'na_na', 'k_k', 'rb_rb', 'cs_cs']
ION_WATER = ['h2o_f', 'h2o_cl', 'h2o_br', 'h2o_i', 'h2o_li', 'h2o_na', 'h2o_k', 'h2o_cs']


def pol_rmse(ff, data, use_sat):
    system = ff.parametrize(Topology.fromOpenmm(data.top), batch=True,
                            use_dipole_saturation=use_sat)
    res = system.getEnergy(data.coords, energy_in_kcal=True, include_bonded=False)
    d = res['pol'].detach() - data.energies['pol']
    return torch.sqrt(torch.mean(d ** 2)).item(), torch.max(torch.abs(d)).item()


def run_set(ff, root, names, title):
    print(f'\n=== {title} (pol, kcal/mol) ===')
    print(f'{"pair":10s} {"rmse(sat)":>10s} {"rmse(none)":>10s} {"worst(sat)":>10s} {"worst(none)":>11s}')
    for name in names:
        pdb = os.path.join(root, f'{name}_scan.pdb')
        csv = os.path.join(root, f'{name}_scan.csv')
        if not os.path.exists(pdb):
            print(f'{name:10s} (missing, skipped)')
            continue
        data = EdaData.from_csv_file(pdb, csv)
        r_sat, w_sat = pol_rmse(ff, data, True)
        r_off, w_off = pol_rmse(ff, data, False)
        print(f'{name:10s} {r_sat:10.3f} {r_off:10.3f} {w_sat:10.3f} {w_off:11.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(__file__), '../../CMM_Data'))
    ap.add_argument('--ff', default=os.path.join(os.path.dirname(__file__), 'ion_water_saturation.xml'))
    args = ap.parse_args()

    ff = ForceFieldXML(args.ff, device='cpu', float_dtype=torch.float64)
    run_set(ff, os.path.join(args.data, 'ion_water', 'ion_ion_scans'),
            LIKE_CHARGE, 'held-out like-charge ion-ion scans')
    run_set(ff, os.path.join(args.data, 'ion_water', 'ion_water_scans'),
            ION_WATER, 'held-out ion-water scans')


if __name__ == '__main__':
    main()
