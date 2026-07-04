import os, sys
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Dict, Callable, List, Literal, Optional, Union
from .base import Logger
from .data import DipoleData, EdaData, EspData, PolarizabilityData
from ..ffxml import ForceFieldXML
from ..units import EV2KCAL, BOHR2ANG, HARTREE2KCAL
from ..topology import Topology


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
                param_mask = torch.tensor(mask, device=ff.device).reshape(-1, 1)
            else:
                param_mask = torch.tensor(mask, device=ff.device)
            
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

    def print_params(self):
        print("===== Optimizing Parameters =====")
        for name, param in self.opt_params.items():
            print(name, param)
        print("===== Parameter Gradient =====")
        for name, param in self.opt_params.items():
            print(name, param.grad)


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


def boltzmann_weight(energies: torch.Tensor):
    kbT = 8.314 * 298.15 / 1000
    return torch.exp(-energies/kbT)


def read_ase_log(logfile):
    data = []
    with open(logfile) as f:
        f.readline()
        for line in f:
            content = line.split()
            if len(content) == 5: # no time
                data.append(tuple(map(float, line.split()[1:])))
            else:
                data.append(tuple(map(float, line.split()[1:-1])))

    if len(data[0]) == 4:
        return pd.DataFrame(data, columns=['time', 'temperature', 'energy', 'density'])
    else:
        return pd.DataFrame(data, columns=['time', 'temperature', 'epot', 'etot', 'density'])


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
        elif weight_func == 'boltzmann':
            self.weight_func = boltzmann_weight
        else:
            raise NotImplementedError(f"Unsupported weighting: {weight_func}")

        self.imbalance_loss = imbalance_loss

    def evaluate(self, data, system=None, **kwargs):
        if system is None:
            system = self.ff.parametrize(Topology.fromOpenmm(data.topology), batch=True, **kwargs)
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
    
    def train(self, datas, num_epoch: int = 10, use_batch: bool = False, **kwargs):
        systems = [self.ff.parametrize(Topology.fromOpenmm(data.topology), batch=True, **kwargs) for data in datas]
        losses = [[] for _ in range(len(datas))]

        for n in range(num_epoch):
            
            self.optimizer.zero_grad()
            
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

                if use_batch:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                else:
                    total_loss.backward()

            if not use_batch:
                self.optimizer.step()
                    

        return losses

    def print_params(self):
        self.optimizer.print_params()

    def compute_density_loss_and_gradient(
        self,
        pdbs: List[os.PathLike],
        trajs: List[os.PathLike],
        ref_densities: List[float],
        temperatures: List[float],
        weights: Union[float, List[float]] = 1.0,
        index: str = ':',
        **kwargs
    ):
        from ase.io import read
        from ase.units import kB

        grads = {key: torch.zeros_like(value) for key, value in self.optimizer.opt_params.items()}
        device = self.ff.device

        weights = [weights for _ in range(len(pdbs))] if isinstance(weights, float) else weights
        calc_densities = []
        for n, (pdb, traj_path, ref_d, temp, weight) in enumerate(zip(pdbs, trajs, ref_densities, temperatures, weights, strict=True)):
            top = Topology.fromPDB(pdb, device)
            system = self.ff.parametrize(top, batch=False, expand_parametrizers_during_init=False, **kwargs)
            traj = read(traj_path, index=index)
            param_grad = {key: torch.zeros_like(value) for key, value in self.optimizer.opt_params.items()}
            density_param_grad = {key: torch.zeros_like(value) for key, value in self.optimizer.opt_params.items()}

            densities = []
            for atoms in tqdm(traj, total=len(traj), desc=f'Processing {n}', disable=not sys.stdout.isatty()):
                coords = torch.tensor(atoms.get_positions() / BOHR2ANG, device=device, requires_grad=False)
                box = torch.tensor(atoms.get_cell().array / BOHR2ANG, device=device, requires_grad=False)
                energy = system.getEnergy(coords, box)['total'] * HARTREE2KCAL
                energy.backward()
                d = np.sum(atoms.get_masses()) * 1.66053906892  / atoms.get_volume()
                densities.append(d)

                for key in self.optimizer.opt_params:
                    g = self.optimizer.opt_params[key].grad
                    with torch.no_grad():
                        param_grad[key].add_(g)
                        density_param_grad[key].add_(g * d)
                    g.zero_()
            
            avg_d = np.mean(densities)
            calc_densities.append(avg_d)
            beta = 1 / (temp * kB * EV2KCAL)

            loss = 0.5*(avg_d-ref_d)**2
            print(f"Density Calculated: {avg_d:.5f}, Reference: {ref_d:.5f}, Error: {avg_d-ref_d:.5f}, L2 Loss: {loss}, Weighted L2 Loss: {loss*weight}")
            for key in self.optimizer.opt_params:
                grad = -beta * (density_param_grad[key] / len(traj) - param_grad[key] * (avg_d / len(traj)) ) * (avg_d - ref_d) * weight
                grads[key].add_(grad)
        
        for key in grads:
            self.optimizer.opt_params[key].grad += grads[key] / len(pdbs)
        
        return np.array(calc_densities)
    
    def compute_heat_of_vap_loss_and_gradient(
        self,
        liq_pdbs: List[os.PathLike],
        liq_trajs: List[os.PathLike],
        gas_pdbs: List[os.PathLike],
        gas_trajs: List[Union[os.PathLike, float, None]],
        ref_hs: List[float],
        temperatures: List[float],
        weights: Union[float, List[float]] = 1.0,
        index: str = ':',
        **kwargs
    ):
        from ase.io import read
        from ase.units import kB

        grads = {key: torch.zeros_like(value) for key, value in self.optimizer.opt_params.items()}
        device = self.ff.device

        weights = [weights for _ in range(len(ref_hs))] if isinstance(weights, float) else weights
        calc_hs = []
        for n in range(len(liq_pdbs)):
            liq_top = Topology.fromPDB(liq_pdbs[n], device)
            liq_system = self.ff.parametrize(liq_top, batch=False, expand_parametrizers_during_init=False, **kwargs)
            liq_traj = read(liq_trajs[0], index=index)
            liq_du_dparam = {key: torch.zeros_like(value) for key, value in self.optimizer.opt_params.items()}
            liq_u_du_dparam = {key: torch.zeros_like(value) for key, value in self.optimizer.opt_params.items()}

            u_liq = []
            t_liq = []
            for atoms in tqdm(liq_traj, total=len(liq_traj), desc=f'Processing liq-phase {n}', disable=not sys.stdout.isatty()):
                coords = torch.tensor(atoms.get_positions() / BOHR2ANG, device=device, requires_grad=False)
                box = torch.tensor(atoms.get_cell().array / BOHR2ANG, device=device, requires_grad=False)
                energy = liq_system.getEnergy(coords, box)['total'] * HARTREE2KCAL
                energy.backward()

                u = energy.item()
                u_liq.append(u)
                t_liq.append(atoms.get_temperature())
                for key in self.optimizer.opt_params:
                    g = self.optimizer.opt_params[key].grad
                    with torch.no_grad():
                        liq_du_dparam[key].add_(g)
                        liq_u_du_dparam[key].add_(g * u)
                    g.zero_()
            
            beta = 1 / (temperatures[n] * kB * EV2KCAL)
            u_liq_avg = np.mean(u_liq)
            t_liq_avg = np.mean(t_liq)
            liq_duavg_dparam = {}
            for key in grads:
                liq_duavg_dparam[key] = (1 + beta * u_liq_avg) * liq_du_dparam[key] / len(liq_traj) - beta * liq_u_du_dparam[key] / len(liq_traj)
            
            gas_top = Topology.fromPDB(gas_pdbs[n], device)
            num_mols = liq_top.natoms // gas_top.natoms
            
            # Gas-phase
            gas_duavg_dparam = {}
            t_gas_avg = None
            if isinstance(gas_trajs[n], float):
                u_gas_avg = gas_pdbs[n]
            elif str(gas_trajs[n]).endswith('.log'):
                df = read_ase_log(gas_trajs[n])
                u_gas_avg = np.mean(df['epot'].values) * EV2KCAL
                t_gas_avg = np.mean(df['temperature'].values)
            else:
                gas_system = self.ff.parametrize(gas_top, batch=False, expand_parametrizers_during_init=False, **kwargs)
                gas_traj = read(gas_trajs[n], index=index)
                u_gas = []
                t_gas = []
                gas_du_dparam = {key: torch.zeros_like(value) for key, value in self.optimizer.opt_params.items()}
                gas_u_du_dparam = {key: torch.zeros_like(value) for key, value in self.optimizer.opt_params.items()}
                for atoms in tqdm(gas_traj, total=len(gas_traj), desc=f'Processing gas-phase {n}', disable=not sys.stdout.isatty()):
                    coords = torch.tensor(atoms.get_positions() / BOHR2ANG, device=device, requires_grad=False)
                    box = torch.tensor(atoms.get_cell().array / BOHR2ANG, device=device, requires_grad=False)
                    energy = gas_system.getEnergy(coords, box)['total'] * HARTREE2KCAL
                    energy.backward()

                    u = energy.item()
                    u_gas.append(u)
                    t_gas.append(atoms.get_temperature())
                    for key in self.optimizer.opt_params:
                        g = self.optimizer.opt_params[key].grad
                        with torch.no_grad():
                            gas_du_dparam[key].add_(g)
                            gas_u_du_dparam[key].add_(g * u)
                        g.zero_()
                
                u_gas_avg = np.mean(u_gas)
                t_gas_avg = np.mean(t_gas)
                for key in grads:
                    gas_duavg_dparam[key] = (1 + beta * u_gas_avg) * gas_du_dparam[key] / len(gas_traj) - beta * gas_u_du_dparam[key] / len(gas_traj)

            # compute heat of vaporization
            dh = u_gas_avg - u_liq_avg / num_mols + kB * EV2KCAL * temperatures[n]
            if t_gas_avg is not None:
                dh -= kB * EV2KCAL * (t_gas_avg - t_liq_avg) * (3 * gas_top.natoms - 6) / 2
            calc_hs.append(dh)
            
            for key in self.optimizer.opt_params:
                if len(gas_duavg_dparam) > 0:
                    grad = (gas_duavg_dparam[key] - liq_duavg_dparam[key] / num_mols) * (dh - ref_hs[n]) * weights[n]
                else:
                    grad = (-liq_duavg_dparam[key] / num_mols)  * (dh - ref_hs[n]) * weights[n]
                grads[key].add_(grad)

            loss = 0.5*(ref_hs[n]-dh)**2
            print(f"Delta H Calculated: {dh:.5f}, Reference: {ref_hs[n]:.5f}, Error: {dh-ref_hs[n]:.5f}, L2 Loss: {loss}, Weighted L2 Loss: {loss*weights[n]}")

        for key in grads:
            self.optimizer.opt_params[key].grad += grads[key] / len(liq_pdbs)
        
        return np.array(calc_hs)

