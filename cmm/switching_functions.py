import torch

def switch_543(dists: torch.Tensor, r_switch: torch.Tensor, r_cutoff: torch.Tensor):
    all_indices = torch.arange(dists.size(0))
    switched_pairs = torch.nonzero(torch.where(dists > r_switch, all_indices, torch.tensor(0.0)), as_tuple=False).flatten()
    x = (dists[switched_pairs] - r_switch) / (r_cutoff - r_switch)
    switch_values = torch.ones_like(dists)
    switch_values[switched_pairs] = switch_values[switched_pairs] - 6 * torch.pow(x, 5) + 15 * torch.pow(x, 4) - 10 * torch.pow(x, 3)
    return switch_values