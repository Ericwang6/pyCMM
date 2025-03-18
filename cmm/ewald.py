import torch
import math

def long_range_potential(coords: torch.Tensor, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor, box: torch.Tensor, alpha: torch.Tensor, max_hkl: torch.NumberType):

    # Reciporcal lattice vectors #
    V = torch.det(box) # volume of box
    reciprocal_box = torch.stack((
        torch.linalg.cross(box[1], box[2]),
        torch.linalg.cross(box[2], box[0]),
        torch.linalg.cross(box[0], box[1])
    )) / V

    # @SPEED: Two optimizations here.
    # 1) only do calculation over positive k vectors and just multiply by 2 (or whatever the factor is)
    # 2) Take this rectangular range and make it spherical by making the cutoff on k^2 not k_max.
    hkl_range = torch.arange(-max_hkl, max_hkl + 1, device=coords.device)
    all_hkl = torch.cartesian_prod(hkl_range, hkl_range, hkl_range).to(torch.float64)
    all_hkl = all_hkl[torch.norm(all_hkl, dim=1) != 0.0]
    kvectors = torch.matmul(all_hkl, reciprocal_box)
    
    # Precalculating gaussian factors
    k_squared = torch.einsum('ij,ij->i', kvectors, kvectors)
    gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / (alpha * alpha)) / k_squared

    # Calculating all structure factors
    k_dot_r = torch.matmul(kvectors, coords.T)
    cos_k_dot_r = torch.cos(2 * torch.pi * k_dot_r)
    sin_k_dot_r = torch.sin(2 * torch.pi * k_dot_r)

    F_l_real = q.expand(kvectors.size(0), -1) - torch.einsum('kj,nij,ki->kn', kvectors, t, kvectors) * (2 * torch.pi) * (2 * torch.pi) / 3
    F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
    
    exp_k_dot_r = torch.complex(cos_k_dot_r, sin_k_dot_r)
    exp_minus_k_dot_r = torch.complex(cos_k_dot_r, -sin_k_dot_r)
    F_2 = torch.complex(F_l_real, F_l_imag)
    structure_factors = torch.sum(F_2 * exp_k_dot_r, dim=1)
    phi_expanded = (gaussian_factors * structure_factors).unsqueeze(1) * exp_minus_k_dot_r
    
    potential = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V) # can take .real inside sum since .imag sums to zero.
    field = 2 * (
        torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
    ) / V
    k_outer = torch.vmap(torch.outer)(kvectors, kvectors).reshape(-1, 9)
    field_grad = 4 * torch.pi * (
        torch.matmul(phi_expanded.T, torch.complex(k_outer, torch.zeros_like(k_outer))).real.reshape(-1, 3, 3)
    ) / V

    # Now add in the self contributions to potential, field, and field gradient #
    alpha_over_root_pi = alpha / torch.sqrt(torch.tensor(torch.pi))
    potential = potential - 2 * alpha_over_root_pi * q
    field = field + alpha_over_root_pi * (4 * alpha * alpha / 3) * p
    field_grad = field_grad + alpha_over_root_pi * (16 * alpha * alpha * alpha * alpha / 5) * t / 3

    return potential, field, field_grad

def long_range_potential_rank_1(coords: torch.Tensor, q: torch.Tensor, p: torch.Tensor, box: torch.Tensor, alpha: torch.Tensor, max_hkl: torch.NumberType):

    # Reciporcal lattice vectors #
    V = torch.det(box) # volume of box
    reciprocal_box = torch.stack((
        torch.linalg.cross(box[1], box[2]),
        torch.linalg.cross(box[2], box[0]),
        torch.linalg.cross(box[0], box[1])
    )) / V

    # @SPEED: Two optimizations here.
    # 1) only do calculation over positive k vectors and just multiply by 2 (or whatever the factor is)
    # 2) Take this rectangular range and make it spherical by making the cutoff on k^2 not k_max.
    hkl_range = torch.arange(-max_hkl, max_hkl + 1, device=coords.device)
    all_hkl = torch.cartesian_prod(hkl_range, hkl_range, hkl_range).to(torch.float64)
    all_hkl = all_hkl[torch.norm(all_hkl, dim=1) != 0.0]
    kvectors = torch.matmul(all_hkl, reciprocal_box)
    
    # Precalculating gaussian factors
    k_squared = torch.einsum('ij,ij->i', kvectors, kvectors)
    gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / (alpha * alpha)) / k_squared

    # Calculating all structure factors
    k_dot_r = torch.matmul(kvectors, coords.T)
    cos_k_dot_r = torch.cos(2 * torch.pi * k_dot_r)
    sin_k_dot_r = torch.sin(2 * torch.pi * k_dot_r)

    F_l_real = q.expand(kvectors.size(0), -1)
    F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
    
    exp_k_dot_r = torch.complex(cos_k_dot_r, sin_k_dot_r)
    exp_minus_k_dot_r = torch.complex(cos_k_dot_r, -sin_k_dot_r)
    F_2 = torch.complex(F_l_real, F_l_imag)
    structure_factors = torch.sum(F_2 * exp_k_dot_r, dim=1)
    phi_expanded = (gaussian_factors * structure_factors).unsqueeze(1) * exp_minus_k_dot_r
    
    potential = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V) # can take .real inside sum since .imag sums to zero.
    field = 2 * (
        torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
    ) / V
    
    # Now add in the self contributions to potential, field, and field gradient #
    alpha_over_root_pi = alpha / torch.sqrt(torch.tensor(torch.pi))
    potential = potential - 2 * alpha_over_root_pi * q
    field = field + alpha_over_root_pi * (4 * alpha * alpha / 3) * p

    return potential, field