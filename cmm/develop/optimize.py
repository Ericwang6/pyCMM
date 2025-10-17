import torch
from typing import Dict, Callable, List, Literal, Optional
from .base import Logger
from .data import DipoleData, EdaData, EspData, PolarizabilityData
from ..ffxml import ForceFieldXML
from ..units import HARTREE2KCAL
from ..topology import Topology
from tqdm import tqdm


class Optimizer:

    def __init__(
        self, 
        ff: ForceFieldXML, 
        freeze_water: bool = True, 
        opt_params: List[str] = list(), 
        freeze_params: List[str] = list(), 
        freeze_types: List[str] = list(), 
        freeze_rule: Literal['any', 'all'] = 'all',
        optim: str = 'adam', 
        lr: float = 0.05, 
        l2: float = 100.0, 
        l2_params = dict(),
        additional_positive_constraints: List[str] = list(),
        fit_pair_params_only: bool = False,
        enforce_iso_pol: List[str] = list(),
        **kwargs
    ):

        # Logging
        self.logger = kwargs.get('logger', None)
        if self.logger is None:
            self.logger = Logger

        # Set Parameters to be Optimized
        self.opt_params: Dict[str, torch.Tensor] = {}
        for force_name in ff.pset.data:
            for item_name in ff.pset.data[force_name]:
                if fit_pair_params_only and item_name != 'Pair':
                    continue
                for param_name in ff.pset.data[force_name][item_name]:
                    p = ff.pset.data[force_name][item_name][param_name]
                    if torch.is_tensor(p) and p.is_floating_point():
                        can_param_name = f'{force_name}/{item_name}/{param_name}'
                        if (can_param_name in opt_params) or (param_name in opt_params):
                            self.opt_params[can_param_name] = p
        
        for param_name in freeze_params:
            remove = None
            if param_name in self.opt_params:
                remove = param_name
            else:
                for p in self.opt_params:
                    if p.split('/')[-1] == param_name:
                        remove = p
                        break
            if remove:
                self.opt_params.pop(remove)
        
        self.logger.info(f"The following parameters are being optimized: \n{'\n'.join(self.opt_params.keys())}")

        # Set AtomTypes to be optimized
        self.freeze_types = freeze_types.copy()
        if freeze_water:
            self.freeze_types.append('ow')
            self.freeze_types.append('hw')
        
        self.masks = {}
        reduce_func = any if freeze_rule == 'any' else all
        self.enforce_iso_pol_indices = []
        for can_param_name in self.opt_params:
            mask = []
            force_name, item_name, param_name = tuple(can_param_name.split('/'))
            types = list(zip(*[ff.pset.data[force_name][item_name][p] for p in ff.pset.data[force_name][item_name].keys() if p.startswith('type')]))
            for index, type in enumerate(types):
                if reduce_func(t in self.freeze_types for t in type) or self.opt_params[can_param_name][index].item() == 0.0:
                    mask.append(False)
                else:
                    mask.append(True)
                if can_param_name == 'Polarization/Pol/alpha_xx' and type[0] in enforce_iso_pol:
                    self.enforce_iso_pol_indices.append(index)
            
            if len(self.opt_params[can_param_name].shape) > 1:
                param_mask = torch.tensor(mask).reshape(-1, 1)
            else:
                param_mask = torch.tensor(mask)
            
            self.masks[can_param_name] = param_mask
        
        # params enforced to be positive
        predefined_positive_constraints = [
            'ChargePenetration/CP/Z', 'ChargePenetration/CP/b_elec', 'ChargePenetration/Pair/b_elec',
            'PauliRepulsion/Pauli/q_pauli', 'PauliRepulsion/Pauli/b_pauli', 'PauliRepulsion/Pair/b_pauli',
            'ExchangePolarization/Xpol/b_xpol', 'ExchangePolarization/Pair/b_xpol',
            'Dispersion/Disp/C6_disp', 'Dispersion/Disp/b_disp', 'Dispersion/Pair/b_disp', 'Dispersion/Pair/C6_disp',
            # 'ChargeTransfer/Direct/q_ct_don', 
            'ChargeTransfer/Direct/b_ct', 'ChargeTransfer/Pair/b_ct', 'ChargeTransfer/Indirect/eps_ct',
            'Polarization/Pol/eta', 'Polarization/Pol/alpha_xx', 'Polarization/Pol/alpha_yy', 'Polarization/Pol/alpha_zz',
            'Polarization/Pol/alpha_damp_exponent', 'Polarization/Pol/alpha_damp_max',
            'Bonds/Bond/r_eq', 'Bonds/Bond/D', 'Bonds/Bond/k_b',
            'Angles/Angle/theta_eq', 'Angles/Angle/k_theta',
            # from here are the equilibrium values in the coupling terms
            # by definition they should be asscoicated with their values and not re-defined in the 
            # terms, but the current codes do this
            'Torsions/Torsion/theta_eq_1', 'Torsions/Torsion/theta_eq_2',
            'Angles/Angle/r_eq_1', 'Angles/Angle/r_eq_2', 
            'AngleAngleCoupling/AngleAngle/theta_eq_1', 'AngleAngleCoupling/AngleAngle/theta_eq_2',
            'TorsionBondCoupling/TorsionBond/r_eq', 'TorsionAngleCoupling/TorsionAngle/theta_eq'
        ]
        self.positive_constraints = []
        for can_param_name in self.opt_params:
            param_name = can_param_name.split('/')[-1]
            if (can_param_name in predefined_positive_constraints) or (can_param_name in additional_positive_constraints) or (param_name in additional_positive_constraints):
                self.positive_constraints.append(can_param_name)
        
        self.logger.info(f"The following parameters are enforced to be positive during optimization: \n{'\n'.join(self.positive_constraints)}")
        
        # set torch optimizer
        self.set_torch_optimizer(optim, lr, **kwargs)

        # set L2 regularization
        self.l2 = l2
        self.l2_params = l2_params
        self.l2_params_tensors = {}
        for can_param_name in self.opt_params:
            param_name = can_param_name.split('/')[-1]
            if (param_name in l2_params) or (can_param_name in l2_params):
                self.l2_params_tensors[can_param_name] = self.opt_params[can_param_name].clone().detach()

    def set_torch_optimizer(self, optim: str = 'adam', lr: float = 0.05, **kwargs):
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(list(self.opt_params.values()), lr=lr, **kwargs)
        elif optim == 'sgd':
            self.optimizer = torch.optim.SGD(list(self.opt_params.values()), lr=lr)
        else:
            raise NotImplementedError()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        for name, param in self.opt_params.items():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
                continue
            if name in self.l2_params_tensors:
                param.grad = param.grad + 2 * self.l2 * (param - self.l2_params_tensors[name])
            param.grad = param.grad * self.masks[name]
        
        if self.enforce_iso_pol_indices:
            grad = (self.opt_params['Polarization/Pol/alpha_xx'].grad[self.enforce_iso_pol_indices] + \
                self.opt_params['Polarization/Pol/alpha_yy'].grad[self.enforce_iso_pol_indices] + \
                self.opt_params['Polarization/Pol/alpha_zz'].grad[self.enforce_iso_pol_indices] ) / 3
            self.opt_params['Polarization/Pol/alpha_xx'].grad[self.enforce_iso_pol_indices] = grad
            self.opt_params['Polarization/Pol/alpha_yy'].grad[self.enforce_iso_pol_indices] = grad
            self.opt_params['Polarization/Pol/alpha_zz'].grad[self.enforce_iso_pol_indices] = grad

        self.optimizer.step()
        for name, param in self.opt_params.items():
            if name not in self.positive_constraints:
                continue
            if torch.any(param < 0):
                mask = (param != 0)
                with torch.no_grad():
                    param.clamp_(min=1e-6)
                    param *= mask
                self.logger.warning(f'Some values in {name} are forcibly set to positive')


def weight_mse(y_true: torch.Tensor, y_pred: torch.Tensor, weights=None):
    if weights is None:
        weights = torch.ones_like(y_true) / y_true.numel()
    else:
        weights = weights / torch.sum(weights)

    loss = torch.sum((y_true - y_pred) ** 2 * weights)
    return loss


def sqrt_weight_func(arr: torch.Tensor):
    return torch.exp(-0.5 * torch.sqrt(arr - arr.min()))


def interaction_weight(arr: torch.Tensor):
    weights = torch.ones_like(arr)
    weights[arr > 0] = torch.exp(-0.5 * torch.sqrt(arr[arr > 0]))
    return weights


def misquitta_weight(energies: torch.Tensor, alpha=0.4, e0=25):
    return torch.exp(-alpha * torch.log(energies / e0) ** 2)


class Trainer:
    def __init__(
        self, 
        ff: ForceFieldXML, 
        optimizer: Optimizer, 
        target_weights = dict(), 
        eda_weights = dict(), 
        qsum_constr: float = 1e5, 
        weight_func: str | Callable | None = 'interaction',
        imbalance_loss: float = 1.0
    ):
        self.ff = ff
        self.optimizer = optimizer
        self.optimize_charge = 'Multipoles/Multipole/c0' in self.optimizer.opt_params

        # weights
        self.target_weights = {
            'EdaData': 1.0,
            'EspData': 1000.0,
            'DipoleData': 1000.0,
            'PolarizabilityData': 1000.0
        }
        self.target_weights.update(target_weights)
        
        self.eda_weights = {
            "perm_elec": 1.0,
            "pauli": 1.0,
            "disp": 1.0,
            "ct": 1.0,
            "pol": 1.0,
            "total": 1.0
        }
        self.eda_weights.update(eda_weights)
        self.qsum_constr = qsum_constr
        
        if weight_func is None:
            self.weight_func = torch.ones_like
        elif isinstance(weight_func, Callable):
            self.weight_func = weight_func
        elif weight_func == 'interaction':
            self.weight_func = interaction_weight
        else:
            raise NotImplementedError(f"Unsupported weighting: {weight_func}")

        self.imbalance_loss = imbalance_loss

    def evaluate(self, data, system=None, **kwargs):
        if system is None:
            system = self.ff.parametrize(Topology.fromOpenmm(data.top), batch=True, **kwargs)
        
        if isinstance(data, EdaData):
            res = system.getEnergy(data.coords, energy_in_kcal=True, include_bonded=False)
            ref = data.energies
            ref_charge, charge = torch.zeros_like(res['charges']), torch.zeros_like(res['charges'])   
        elif isinstance(data, EspData):
            res = system.getEnergy(data.coord.unsqueeze(0), grid=[data.grid])
            res, ref, charge = res['grid_epot'][0], data.esp, res['charges']
            ref_charge = torch.ones_like(charge) * data.charge
        elif isinstance(data, DipoleData):
            res = system.getEnergy(data.coords)
            res, ref, charge = res['dipoles'], data.dipos, res['charges']
            ref_charge = torch.ones_like(charge) * data.charge
        elif isinstance(data, PolarizabilityData):
            res = system.getEnergy(data.coords)
            res, ref, charge = res['polarizability'], data.pol, res['charges']
            ref_charge = torch.ones_like(charge) * data.charge
        else:
            raise NotImplementedError(f"Not supported data type: {type(data)}")
        
        return res, ref, ref_charge, charge
    
    def train(self, datas, num_epoch: int = 10, **kwargs):
        systems = [self.ff.parametrize(Topology.fromOpenmm(data.top), batch=True, **kwargs) for data in datas]
        losses = [[] for _ in range(len(datas))]

        for n in range(num_epoch):

            for i, (data, system) in enumerate(zip(datas, systems)):
                res, ref, ref_charge, charge = self.evaluate(data, system)
                loss_weight = self.target_weights.get(data.__class__.__name__, 1000.0)
                total_loss = 0.0
                if isinstance(data, EdaData):
                    loss = {}
                    for key in ref:
                        if self.imbalance_loss != 1.0:
                            imbalance = torch.ones_like(ref[key], device=ref[key].device, dtype=ref[key].dtype) * self.imbalance_loss
                            weights = torch.where(ref[key] > res[key], imbalance, 1.0) * self.weight_func(ref['total'])
                        else:
                            weights = self.weight_func(ref['total'])
                        l = weight_mse(ref[key], res[key], weights)
                        total_loss += l * loss_weight * self.eda_weights.get(key, 1.0)
                        loss[key] = l.detach().item()
                else:
                    l = weight_mse(ref, res)
                    total_loss += l * loss_weight
                    loss = l.detach().item()
                print(n, data.__class__.__name__, loss)
                losses[i].append(loss)
            
                if self.optimize_charge:
                    qloss = torch.sum((ref_charge - charge) ** 2)
                    total_loss += qloss * self.qsum_constr

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        return losses