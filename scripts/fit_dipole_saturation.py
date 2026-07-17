"""Fit the dipole-saturation parameters of the halide anions against the EDA
polarization energy of the monovalent +/- ion-ion scans.

Staged so the parameters cannot step over each other:
  stage 1: sat_c_iso alone (sat_e0 frozen at 0.05 a.u., anisotropic term off)
  stage 2: sat_c_iso + per-ion sat_e0
  stage 3: sat_c_ani + sat_w alone (isotropic part frozen)
  stage 4: joint fine-tune of all four with small learning rates

Only the `pol` channel enters the loss; free-ion polarizabilities, eta, and
every non-polarization force are frozen.

Usage:
    python scripts/fit_dipole_saturation.py [--data /path/to/CMM_Data] \
        [--epochs1 N] [--epochs2 N] [--out scripts/ion_water_saturation.xml]
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
from cmm.develop.optimize import interaction_weight, weight_mse

CATIONS = ['li', 'na', 'k', 'rb', 'cs']  # divalent Mg/Ca excluded from the fit
DIVALENT = ['mg', 'ca']
ANIONS = ['f', 'cl', 'br', 'i']
PAIRS = [f'{c}_{a}' for c in CATIONS for a in ANIONS]
DIVALENT_PAIRS = [f'{c}_{a}' for c in DIVALENT for a in ANIONS]
HALIDE_TYPES = ['f-', 'cl-', 'br-', 'i-']

E0_MIN, E0_MAX = 0.005, 0.5  # a.u., keep the onset field physically sane


def load_scans(root, pairs):
    datas = []
    for p in pairs:
        data = EdaData.from_csv_file(os.path.join(root, f'{p}_scan.pdb'),
                                     os.path.join(root, f'{p}_scan.csv'))
        data.name = p
        datas.append(data)
    return datas


def halide_mask(ff):
    types = list(ff.pset.find('Polarization/Pol/type'))
    mask = torch.zeros(len(types))
    for t in HALIDE_TYPES:
        mask[types.index(t)] = 1.0
    return mask


def run_stage(ff, datas, param_groups, num_epoch, log_every=25):
    """param_groups: list of (pset_path, lr). Only halide entries train."""
    mask = halide_mask(ff)
    params = []
    for path, lr in param_groups:
        p = ff.pset.find(path)
        p.requires_grad_(True)
        params.append({'params': [p], 'lr': lr})
    optimizer = torch.optim.Adam(params)

    systems = [ff.parametrize(Topology.fromOpenmm(d.top), batch=True) for d in datas]
    for n in range(num_epoch):
        epoch_loss = 0.0
        for data, system in zip(datas, systems):
            res = system.getEnergy(data.coords, energy_in_kcal=True, include_bonded=False)
            weights = interaction_weight(data.energies['total'])
            loss = weight_mse(data.energies['pol'], res['pol'], weights)
            optimizer.zero_grad()
            loss.backward()
            for g in optimizer.param_groups:
                for p in g['params']:
                    if p.grad is not None:
                        p.grad *= mask
            optimizer.step()
            epoch_loss += loss.item()
        with torch.no_grad():
            for path, _ in param_groups:
                p = ff.pset.find(path)
                if path.endswith('sat_e0'):
                    p.data[mask.bool()] = p.data[mask.bool()].clamp(E0_MIN, E0_MAX)
                elif path.endswith('sat_w'):
                    p.data[mask.bool()] = p.data[mask.bool()].clamp(0.1, 10.0)
                elif path.endswith('sat_c_ani'):
                    # either sign is well-defined (free-ion limit exact, energy
                    # bounded, SPD floor in the solver); negative = extra
                    # axial damping instead of a parallel rise
                    p.data[mask.bool()] = p.data[mask.bool()].clamp(-10.0, 10.0)
                else:
                    p.data[mask.bool()] = p.data[mask.bool()].clamp_min(1e-6)
        if n % log_every == 0 or n == num_epoch - 1:
            print(f'epoch {n:4d}  sum pol loss = {epoch_loss:.4f}')


def report(ff, datas):
    print('\n=== pol RMSE per pair (kcal/mol) ===')
    total_sq, total_n = 0.0, 0
    for data in datas:
        system = ff.parametrize(Topology.fromOpenmm(data.top), batch=True)
        res = system.getEnergy(data.coords, energy_in_kcal=True, include_bonded=False)
        d = (res['pol'].detach() - data.energies['pol'])
        rmse = torch.sqrt(torch.mean(d ** 2)).item()
        total_sq += torch.sum(d ** 2).item()
        total_n += d.numel()
        print(f'{data.name:8s} rmse={rmse:8.3f}  worst={torch.max(torch.abs(d)).item():8.3f}')
    print(f'overall rmse = {(total_sq / total_n) ** 0.5:.3f} kcal/mol')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(__file__), '../../CMM_Data'))
    ap.add_argument('--ff', default=os.path.join(os.path.dirname(__file__), 'ion_water_refit.xml'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'ion_water_saturation.xml'))
    ap.add_argument('--epochs1', type=int, default=100)
    ap.add_argument('--epochs2', type=int, default=200)
    ap.add_argument('--epochs3', type=int, default=150)
    ap.add_argument('--epochs4', type=int, default=150)
    args = ap.parse_args()

    root = os.path.join(args.data, 'ion_water', 'ion_ion_scans')
    datas = load_scans(root, PAIRS)
    datas_divalent = load_scans(root, DIVALENT_PAIRS)
    print(f'loaded {len(datas)} monovalent +/- scans, {sum(d.num for d in datas)} frames total '
          f'({len(datas_divalent)} divalent scans held out)')

    ff = ForceFieldXML(args.ff, device='cpu', float_dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        ff.pset.find('Polarization/Pol/sat_c_ani').zero_()  # isotropic-only

    print('\n===== stage 1: sat_c_iso only (sat_e0 = 0.05 a.u. fixed) =====')
    run_stage(ff, datas, [('Polarization/Pol/sat_c_iso', 0.05)], args.epochs1)
    report(ff, datas)

    print('\n===== stage 2: sat_c_iso + per-ion sat_e0 =====')
    run_stage(ff, datas, [('Polarization/Pol/sat_c_iso', 0.02),
                          ('Polarization/Pol/sat_e0', 0.002)], args.epochs2)
    report(ff, datas)

    # ---- anisotropic term: seed and fit with the isotropic part frozen ----
    mask = halide_mask(ff).bool()
    with torch.no_grad():
        ff.pset.find('Polarization/Pol/sat_c_ani').data[mask] = 0.1
        ff.pset.find('Polarization/Pol/sat_w').data[mask] = 1.0
    print('\n===== stage 3: sat_c_ani + sat_w (isotropic frozen) =====')
    run_stage(ff, datas, [('Polarization/Pol/sat_c_ani', 0.005),
                          ('Polarization/Pol/sat_w', 0.02)], args.epochs3)
    report(ff, datas)

    print('\n===== stage 4: joint fine-tune (all four, small lr) =====')
    run_stage(ff, datas, [('Polarization/Pol/sat_c_iso', 0.01),
                          ('Polarization/Pol/sat_e0', 0.001),
                          ('Polarization/Pol/sat_c_ani', 0.002),
                          ('Polarization/Pol/sat_w', 0.01)], args.epochs4)
    report(ff, datas)
    print('\n--- divalent pairs (held out, not fit) ---')
    report(ff, datas_divalent)

    types = ff.pset.find('Polarization/Pol/type')
    c_iso = ff.pset.find('Polarization/Pol/sat_c_iso')
    e0 = ff.pset.find('Polarization/Pol/sat_e0')
    c_ani = ff.pset.find('Polarization/Pol/sat_c_ani')
    w = ff.pset.find('Polarization/Pol/sat_w')
    print('\n=== fitted parameters ===')
    for i, t in enumerate(types):
        print(f'{t:6s} sat_c_iso={c_iso[i].item():10.5f} sat_e0={e0[i].item():10.5f} '
              f'sat_c_ani={c_ani[i].item():10.5f} sat_w={w[i].item():10.5f}')

    ff.save(args.out)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
