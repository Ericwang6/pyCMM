import torch
import torch.nn as nn


def switch_543(dists: torch.Tensor, r_switch: torch.Tensor, r_cutoff: torch.Tensor):
    all_indices = torch.arange(dists.size(0), dtype=torch.long, device=dists.device)
    switched_pairs = torch.where(dists > r_switch, all_indices, torch.tensor(-1, dtype=torch.long, device=dists.device))
    switched_pairs = switched_pairs[switched_pairs >= 0]
    x = (dists[switched_pairs] - r_switch) / (r_cutoff - r_switch)
    switch_values = torch.ones_like(dists)
    switch_values[switched_pairs] = switch_values[switched_pairs] - 6 * torch.pow(x, 5) + 15 * torch.pow(x, 4) - 10 * torch.pow(x, 3)
    return switch_values


def smooth_function(x):
    return 1 - 10 * x**3 + 15 * x**4 - 6 * x**5


#@torch.compile
class SwitchFunction(nn.Module):
    def __init__(self, on: bool, cutoff: float, buffer: float):
        super().__init__()
        self.on = on
        self.cutoff = cutoff
        self.buffer = buffer
    
    def forward(self, dists: torch.Tensor):
        if self.on:
            return torch.clamp(smooth_function((dists - self.cutoff + self.buffer) / self.buffer), min=0.0, max=1.0)
        else:
            return torch.ones_like(dists, dtype=dists.dtype, device=dists.device)

    