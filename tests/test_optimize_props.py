import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
from cmm.develop.optimize import Trainer, Optimizer
from cmm.ffxml import ForceFieldXML

device = 'cuda'
torch.set_default_dtype(torch.float64)
data_path = os.path.join(os.path.dirname(__file__), 'data')

# ff_path = '/pscratch/sd/e/eric6/pycmm-dev/optimization/properties/test.xml'
# traj_path = '/pscratch/sd/e/eric6/pycmm-dev/workspace_new/water_aalim_1028/water_rdf/npt_298K.traj'

# ff_path = '/pscratch/sd/e/eric6/pycmm-dev/optimization/properties_pauli/all/iter19/forcefield.xml'
# traj_path = '/pscratch/sd/e/eric6/pycmm-dev/optimization/properties_pauli/all/iter19/water/prod.traj'

ff_path = '/pscratch/sd/e/eric6/pycmm-dev/workspace_water/water_1.xml'
traj_path = '/pscratch/sd/e/eric6/pycmm-dev/workspace_water/1_298_15/prod.traj'


def test_density():
    ff = ForceFieldXML(ff_path, device=device, requires_grad=True)
    optim = Optimizer(
        ff, freeze_water=False, 
        opt_params=['C6_disp', 'b_disp', 'q_pauli', 'b_pauli', 'Kdipo_pauli', 'Kquad_pauli', 'Z', 'b_elec', 'q_xpol', 'b_xpol', 'j_cf_pauli']
    )
    
    trainer = Trainer(ff, optim)
    trainer.compute_density_loss_and_gradient(
        [os.path.join(data_path, 'water_216.pdb')],
        [traj_path],
        [0.997],
        [298.15],
        index='::2'
    )
    for key in optim.opt_params:
        print(key, optim.opt_params[key].grad)


def test_heat_of_vap():
    ff = ForceFieldXML(ff_path, device=device, requires_grad=True)
    optim = Optimizer(
        ff, freeze_water=False, 
        opt_params=['C6_disp', 'b_disp', 'q_pauli', 'b_pauli', 'Kdipo_pauli', 'Kquad_pauli', 'Z', 'b_elec', 'q_xpol', 'b_xpol', 'j_cf_pauli']
    )
    
    trainer = Trainer(ff, optim)
    trainer.compute_heat_of_vap_loss_and_gradient(
        [os.path.join(data_path, 'water_216.pdb')],
        [traj_path],
        [os.path.join(data_path, 'water.pdb')],
        [os.path.join(data_path, 'water_gas.log')],
        [10.51],
        [298.15],
        index='::2'
    )
    for key in optim.opt_params:
        print(key, optim.opt_params[key].grad)