import torch
import numpy as np
from scipy.optimize import minimize

class OptimizationDriver:
    def __init__(self,
                 system,
                 method='L-BFGS-B',
                 tolerance: float=1e-6,
                 optimize_coords=True,
                 optimize_box=False
        ) -> None:
        self.system = system
        self.method = method
        self.tolerance = tolerance
        self.optimize_coords = optimize_coords
        self.optimize_box = optimize_box
        if self.optimize_coords == False and self.optimize_box == False:
            raise ValueError("Both optimize_coords and optimize_box are set to False. At least one must be True to run.")
        
    def energy_and_gradient(self, coords_and_box_flat: np.ndarray):
        coords_flat = coords_and_box_flat[:-9]
        box_flat = coords_and_box_flat[-9:]
        coords = torch.from_numpy(coords_flat.reshape((-1, 3))).requires_grad_(self.optimize_coords).to(self.device)
        box = torch.from_numpy(box_flat.reshape((3, 3))).requires_grad_(self.optimize_box).to(self.device)

        energies = self.system.getEnergy(coords, box)
        energies['total'].backward()
        total_grads = np.zeros_like(coords_and_box_flat)
        if coords.grad is not None:
            total_grads[:-9] += coords.grad.numpy().flatten()
        if box.grad is not None:
            total_grads[-9:] += box.grad.numpy().flatten()
        return energies['total'].item(), total_grads

    def run(self, coords: torch.Tensor, box: torch.Tensor):
        self.device = coords.device
        combined_coords = np.hstack((coords.detach().numpy().flatten(), box.detach().numpy().flatten()))
        result = minimize(self.energy_and_gradient, combined_coords, method=self.method, tol=self.tolerance, jac=True, options={'disp': True, 'maxiter': 1000})
        coords_opt_flat = result.x[:-9]
        box_opt_flat = result.x[-9:]
        coords_opt = torch.tensor(coords_opt_flat.reshape((-1, 3)), dtype=coords.dtype, requires_grad=False)
        box_opt = torch.tensor(box_opt_flat.reshape((3, 3)), dtype=box.dtype, requires_grad=False)
        return coords_opt, box_opt, result