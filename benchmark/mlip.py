import sys, os, glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cmm.develop.data import EdaData
from cmm.units import EV2KCAL
from cmm.develop.metrics import plot_correlation


def get_eda_out(dirname, k_thresh=None):
    files = []
    for file in glob.glob(os.path.join(dirname, '*/eda.out')):
        group = os.path.basename(os.path.dirname(os.path.dirname(file)))
        if group.startswith('_') or group.startswith('md_2.6') or group.startswith('md_2.7'):
            continue
        basename = os.path.basename(os.path.dirname(file))
        if basename.startswith('k') and k_thresh is not None:
            if int(basename.split('_')[1]) < k_thresh:
                continue
        if os.path.isfile(os.path.join(os.path.dirname(file), 'done.tag')):
            files.append(file)
    return files

    
def mk_uma_model(model, device):
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    calc = FAIRChemCalculator(predictor, task_name="omol")
    return calc


def mk_aimnet_model(model, device):
    from aimnet2calc import AIMNet2ASE
    calc = AIMNet2ASE(model)
    return calc


def mk_mace_model(model, device):
    from mace.calculators import mace_omol, mace_mp, mace_off
    
    model_type = model.split('/')[0]
    model_name = model[len(model_type)+1:]

    if model_type == 'mace_omol':
        calc = mace_omol(model=model_name, device=device)
    elif model_type == 'mace_off':
        calc = mace_off(model=model_name, device=device)
    elif model_type == 'mace_mp':
        calc = mace_mp(model=model_name, device=device)
    else:
        raise NotImplementedError(f'Invalid model type: {model_type}')

    return calc


def mk_model(model, device, **kwargs):
    if model.startswith('uma'):
        return mk_uma_model(model, device, **kwargs)
    elif model.startswith('mace'):
        return mk_mace_model(model, device, **kwargs)
    elif model.startswith('aimnet'):
        return mk_aimnet_model(model, device, **kwargs)
    else:
        raise NotImplementedError(f"Invalide model: {model}")
        

    
model = 'uma-s-1p1'
# model = 'mace_off/medium'
# model = 'mace_omol/extra_large'
# model = 'aimnet2'

files = get_eda_out('/pscratch/sd/e/eric6/pycmm-dev/optimization/trimers/methanol_trimer')
device = 'cuda'
output_dir = 'methanol_trimer_uma'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


calc = mk_model(model, device)
data = EdaData.from_files(None, files, use_cls_disp=False)
data = data[data.energies['total'] < 5]

ml_energies = []
for molecule, frags in tqdm(data.get_ase_atoms()):
    molecule.calc = calc
    if hasattr(calc, 'set_charge'):
        calc.set_charge(molecule.info['charge'])
        calc.set_mult(molecule.info['spin'])    
    total_energy = molecule.get_potential_energy()
    frags_energy = []
    for frag in frags:
        frag.calc = calc
        if hasattr(calc, 'set_charge'):
            calc.set_charge(frag.info['charge'])
            calc.set_mult(frag.info['spin'])
        frags_energy.append(frag.get_potential_energy())
    total_energy -= sum(frags_energy)
    total_energy *= EV2KCAL
    ml_energies.append(total_energy)

ml_energies = np.array(ml_energies).flatten()
qm_energies = data.energies['total'].numpy(force=True)

df = zip(files, ml_energies, qm_energies)
df = pd.DataFrame(df, columns=['file', model, 'QM'])
df.to_csv(os.path.join(output_dir, 'result.csv'))

fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), constrained_layout=True)
plot_correlation(qm_energies, ml_energies, xlabel='QM (kcal/mol)', ylabel=f'{model} (kcal/mol)', ax=ax)
fig.savefig(os.path.join(output_dir, 'result.pdf'))