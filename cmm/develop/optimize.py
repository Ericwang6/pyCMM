import torch
from typing import Dict, Callable
from .base import Logger
from .data import DipoleData, EdaData, EspData, PolarizabilityData
from ..forcefield import CMMForceField



class Optimizer:
    def __init__(
        self, 
        ff: CMMForceField, 
        freeze_water: bool = True, 
        opt_params = None, 
        freeze_params = dict(), 
        freeze_types = dict(), 
        optim: str = 'adam', 
        lr: float = 0.05, 
        l2: float = 100.0, 
        l2_params = dict(),
        **kwargs
    ):
        self.params = []
        self.masks = []
        self.names = []

        # don't optimize water
        if freeze_water:
            self.freeze_types = {
                "atomic_params": ['ow', 'hw'],
                "pair_params": [('ow', 'hw')],
                "bond_params": [('ow', 'hw')],
                "angle_params": [("hw", "ow", "hw")]
            }
        else:
            self.freeze_types = {}
        
        for param_name in freeze_types:
            ftypes = self.freeze_types.get(param_name, []) + freeze_types[param_name]
            self.freeze_types[param_name] = ftypes
        
        if opt_params:
            self.opt_params = opt_params
        else:
            self.opt_params = {param_name: [key for key in ff.params[param_name]] for param_name in ff.params}
            for param_name in freeze_params:
                for p in freeze_params[param_name]:
                    self.opt_params[param_name].remove(p)

        for param_name in self.opt_params:
            freeze_types = self.freeze_types.get(param_name, [])

            # not in free types - mask is true - allow to optimize
            mask = []
            for t in ff.params[param_name]['type']:
                t = t if isinstance(t, str) else tuple(t)
                mask.append(t not in freeze_types)

            params_to_opt = self.opt_params.get(param_name, [])
            
            for key in params_to_opt:
                if not isinstance(ff.params[param_name][key], torch.Tensor):
                    continue
                if ff.params[param_name][key].dtype is torch.long:
                    continue
                
                if len(ff.params[param_name][key].shape) > 1:
                    param_mask = torch.tensor(mask).reshape(-1, 1)
                else:
                    param_mask = torch.tensor(mask)

                ff.params[param_name][key].requires_grad = True
                self.params.append(ff.params[param_name][key])
                self.masks.append((ff.params[param_name][key] != 0.0) * param_mask)
                self.names.append(f'{param_name}/{key}')
        
        # Logging
        self.logger = kwargs.get('logger', None)
        if self.logger is None:
            self.logger = Logger
        
        # params enforced to be positive
        positive_constraints = {
            "atomic_params": ['Z', 'b_elec', 'b_pauli', 'C6_disp', 'b_disp', 'alpha', 'eta', 'b_xpol', 'b_ct', 'q_pauli'],
            "pair_params": ["eps_ct"],
            "bond_params": ['r_eq'],
            "angle_params": ['theta_eq']
        }
        self.positive_constraints = set()
        for key in positive_constraints:
            for val in positive_constraints[key]:
                self.positive_constraints.add(f'{key}/{val}')
        
        # set torch optimizer
        self.set_torch_optimizer(optim, lr, **kwargs)

        # set L2 regularization
        self.l2 = l2
        self.l2_params = l2_params
        self.l2_params_tensors = {}
        for param_name in ff.params:
            plist = self.l2_params.get(param_name, [])
            for key in ff.params[param_name]:
                if key in plist:
                    self.l2_params_tensors[f'{param_name}/{key}'] = ff.params[param_name][key].clone().detach()

    @property
    def named_params(self) -> Dict[str, torch.Tensor]:
        return {name: param for name, param in zip(self.names, self.params)}

    def set_torch_optimizer(self, optim: str = 'adam', lr: float = 0.05, **kwargs):
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=lr, **kwargs)
        elif optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=lr)
        else:
            raise NotImplementedError()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        for name, param, mask in zip(self.names, self.params, self.masks):
            if param.grad is None:
                param.grad = torch.zeros_like(param)
                continue
            if name in self.l2_params_tensors:
                param.grad = param.grad + 2 * self.l2 * (param - self.l2_params_tensors[name])
            param.grad = param.grad * mask
            # if name == 'pair_params/eps_ct':
            #     print(name, param.grad)
        self.optimizer.step()
        for name, param in self.named_params.items():
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
    return torch.exp(-torch.sqrt(arr - arr.min()))


def interaction_weight(arr: torch.Tensor):
    weights = torch.ones_like(arr)
    weights[arr > 0] = torch.exp(-0.5 * torch.sqrt(arr[arr > 0]))
    return weights


def misquitta_weight(energies: torch.Tensor, alpha=0.4, e0=25):
    return torch.exp(-alpha * torch.log(energies / e0) ** 2)


class Trainer:
    def __init__(self, ff: CMMForceField, optimizer: Optimizer, target_weights = dict(), eda_weights = dict(), qsum_constr: float = 1e5, weight_func: str = 'interaction'):
        self.ff = ff
        self.optimizer = optimizer

        # weights
        self.target_weights = {
            EdaData: 1.0,
            EspData: 1000.0,
            DipoleData: 1000.0,
            PolarizabilityData: 1000.0
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

    def evaluate(self, datas, systems=None):
        if systems is None:
            systems = [self.ff.parametrize(data.top) for data in datas]
        
        results, refs, ref_charges, charges = [], [], [], []
        for data, system in zip(datas, systems):
            if isinstance(data, EdaData):
                res = system.batch_evalute(data.coords)
                ref = data.energies
                ref_charge, charge = torch.tensor(0.0), torch.tensor(0.0)   
            elif isinstance(data, EspData):
                prop = system.evaluate_electric_properties(data.coord, data.grid)
                res, ref, charge = prop['esp'], data.esp, prop['charge'].flatten()
                ref_charge = torch.ones_like(charge).flatten() * data.charge
            elif isinstance(data, DipoleData):
                prop = system.batch_evaluate_electric_properties(data.coords)
                res, ref, charge = prop['dipole'], data.dipos, prop['charge'].flatten()
                ref_charge = torch.ones_like(charge).flatten() * data.charge
            elif isinstance(data, PolarizabilityData):
                prop = system.batch_evaluate_electric_properties(data.coords)
                res, ref, charge = prop['polarizability'], data.pol, prop['charge'].flatten()
                ref_charge = torch.ones_like(charge).flatten() * data.charge
            else:
                raise NotImplementedError(f"Not supported data type: {type(data)}")
            
            results.append(res)
            refs.append(ref)
            ref_charges.append(ref_charge)
            charges.append(charge)
        ref_charges = torch.hstack(ref_charges)
        charges = torch.hstack(charges)
        
        return results, refs, ref_charges, charges
    
    def train(self, datas, num_epoch: int = 10, data_weights=None):
        systems = [self.ff.parametrize(data.top) for data in datas]
        losses = [[] for _ in range(len(datas))]
        
        if data_weights is None:
            data_weights = [float(getattr(data, 'num', 1.0)) for data in datas]
        data_weights = torch.tensor(data_weights)
        data_weights /= torch.sum(data_weights)

        for n in range(num_epoch):
            results, refs, ref_charges, charges = self.evaluate(datas, systems)
            total_loss = 0.0
            for i, (res, ref, data) in enumerate(zip(results, refs, datas)):
                loss_weight = self.target_weights.get(data.__class__, 1000.0) * data_weights[i]
                # eda data
                if isinstance(res, dict):
                    loss = {}
                    for key in ref:
                        l = weight_mse(ref[key], res[key], interaction_weight(ref['total']))
                        total_loss += l * loss_weight * self.eda_weights.get(key, 1.0)
                        loss[key] = l.detach().item()
                else:
                    l = weight_mse(ref, res)
                    total_loss += l * loss_weight
                    loss = l.detach().item()
                print(n, data.__class__.__name__, loss)
                losses[i].append(loss)
            
            if 'atomic_params/mono' in self.optimizer.named_params:
                qloss = torch.sum((ref_charges - charges) ** 2)
                total_loss += qloss * self.qsum_constr

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return losses
    
    def train_with_batch(self, datas, num_epoch: int = 10, batch_size=1, print_params=None, data_weights=None):
        print("Train with batching...")
        for n in range(num_epoch):
            print(f"# Epoch {n}")
            for i in range(0, len(datas), batch_size):
                batch_data = datas[i: i + batch_size]
                self.train(batch_data, 1, None)
                if print_params is not None:
                    for name in print_params:
                        print(self.optimizer.named_params[name])

class Trainer2:
    def __init__(self, ff: CMMForceField, optimizer: Optimizer, target_weights = dict(), eda_weights = dict(), qsum_constr: float = 1e5, weight_func: str = 'interaction'):
        self.ff = ff
        self.optimizer = optimizer

        # weights
        self.target_weights = {
            EdaData: 1.0,
            EspData: 1000.0,
            DipoleData: 1000.0,
            PolarizabilityData: 1000.0
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

    def evaluate(self, datas, systems=None):
        if systems is None:
            systems = [self.ff.parametrize(datas.topologies[i]) for i in range(len(datas.topologies))]
        
        results = [systems[i].evaluate(datas.coords[i]) for i in range(len(datas.coords))]
        res = {}

        for key in results[0].keys():
            # Extract all tensors for this key and concatenate them
            energy_tensors = [d[key].unsqueeze(0) for d in results]
            res[key] = torch.cat(energy_tensors, dim=0)
            if key in datas.energies:
                assert res[key].size() == datas.energies[key].size()

        return res, datas.energies
    
    def train(self, datas, num_epoch: int = 10, data_weights=None):
        systems = [self.ff.parametrize(datas.topologies[i]) for i in range(len(datas.topologies))]
        losses = [[] for _ in range(len(datas))]
        
        if data_weights is None:
            data_weights = [float(getattr(data, 'num', 1.0)) for data in datas]
        data_weights = torch.tensor(data_weights)
        data_weights /= torch.sum(data_weights)

        for n in range(num_epoch):
            results, refs, ref_charges, charges = self.evaluate(datas, systems)
            total_loss = 0.0
            for i, (res, ref, data) in enumerate(zip(results, refs, datas)):
                loss_weight = self.target_weights.get(data.__class__, 1000.0) * data_weights[i]
                # eda data
                if isinstance(res, dict):
                    loss = {}
                    for key in ref:
                        l = weight_mse(ref[key], res[key], interaction_weight(ref['total']))
                        total_loss += l * loss_weight * self.eda_weights.get(key, 1.0)
                        loss[key] = l.detach().item()
                else:
                    l = weight_mse(ref, res)
                    total_loss += l * loss_weight
                    loss = l.detach().item()
                print(n, data.__class__.__name__, loss)
                losses[i].append(loss)
            
            if 'atomic_params/mono' in self.optimizer.named_params:
                qloss = torch.sum((ref_charges - charges) ** 2)
                total_loss += qloss * self.qsum_constr

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return losses
    
    def train_with_batch(self, datas, num_epoch: int = 10, batch_size=1, print_params=None, data_weights=None):
        print("Train with batching...")
        for n in range(num_epoch):
            print(f"# Epoch {n}")
            for i in range(0, len(datas), batch_size):
                batch_data = datas[i: i + batch_size]
                self.train(batch_data, 1, None)
                if print_params is not None:
                    for name in print_params:
                        print(self.optimizer.named_params[name])