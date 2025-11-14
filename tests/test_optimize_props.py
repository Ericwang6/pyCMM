import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
from cmm.develop.optimize import Trainer, Optimizer
from cmm.ffxml import ForceFieldXML


def test_density():
    device = 'cuda'
    torch.set_default_dtype(torch.float64)
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    ff = ForceFieldXML(os.path.join(data_path, 'water_class.xml'), device=device, requires_grad=True)
    optim = Optimizer(ff, freeze_water=False, opt_params=['C6_disp', 'b_disp'])
    
    trainer = Trainer(ff, optim)
    trainer.compute_density_loss_and_gradient(
        [os.path.join(data_path, 'water_216.pdb')],
        ['/pscratch/sd/e/eric6/pycmm-dev/workspace/water_joe_original_mc/npt_298K_2000ps.traj'],
        [0.997],
        [298.15],
        index='::50'
    )
    for key in optim.opt_params:
        print(key, optim.opt_params[key].grad)


def test_heat_of_vap():
    device = 'cuda'
    torch.set_default_dtype(torch.float64)
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    ff = ForceFieldXML(os.path.join(data_path, 'water_class.xml'), device=device, requires_grad=True)
    optim = Optimizer(ff, freeze_water=False, opt_params=['C6_disp', 'b_disp'])
    
    trainer = Trainer(ff, optim)
    trainer.compute_heat_of_vap_loss_and_gradient(
        [os.path.join(data_path, 'water_216.pdb')],
        ['/pscratch/sd/e/eric6/pycmm-dev/workspace/water_joe_original_mc/npt_298K_2000ps.traj'],
        [os.path.join(data_path, 'water.pdb')],
        [os.path.join(data_path, 'water_gas.log')],
        [10.51],
        [298.15],
        index='::50'
    )
    for key in optim.opt_params:
        print(key, optim.opt_params[key].grad)