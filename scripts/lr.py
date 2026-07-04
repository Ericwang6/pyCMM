import torch
import numpy as np
import time
import torch.nn as nn


def timeit(func):
    def wrapper(*args, **kwargs):
        for _ in range(100):
            func(*args, **kwargs)
        
        start = time.time()
        for _ in range(500):
            func(*args, **kwargs)
        end = time.time()
        print("Time: {:.4f} ms".format((end - start) * 1000 / 500))
    return wrapper


# @timeit
# def long_range_potential(coords: torch.Tensor, q: torch.Tensor, p: torch.Tensor, 
#                            t: torch.Tensor, box: torch.Tensor, alpha: torch.Tensor, max_hkl: torch.NumberType):
#     """
#     Parameters
#     ----------
#     coords: torch.Tensor
#         Atomic coordinates (N, 3)
#     q: torch.Tensor
#         Charges (N,)
#     p: torch.Tensor
#         Dipoles (N, 3)
#     t: torch.Tensor
#         Quadrupoles (N, 3, 3)
#     box: torch.Tensor
#         Box vectors (3, 3)
#     alpha: torch.Tensor
#         Ewald splitting parameter
#     max_hkl: torch.NumberType
#         Maximum h,k,l index values for reciprocal space sum
    
#     Returns
#     -------
#     potential: torch.Tensor
#         Electric potential at each atom (N,)
#     field: torch.Tensor
#         Electric field at each atom (N, 3)
#     field_grad: torch.Tensor
#         Electric field gradient at each atom (N, 3, 3)
#     """
#     # Reciprocal lattice vectors
#     V = torch.det(box)  # volume of box
#     reciprocal_box = torch.stack((
#         torch.linalg.cross(box[1], box[2]),
#         torch.linalg.cross(box[2], box[0]),
#         torch.linalg.cross(box[0], box[1])
#     )) / V

#     # We optimize for h ≥ 0, but keep full range for k and l
#     h_range = torch.arange(0, max_hkl + 1, device=coords.device)
#     kl_range = torch.arange(-max_hkl, max_hkl + 1, device=coords.device)
    
#     # Create all combinations and convert to float64
#     all_hkl = torch.cartesian_prod(h_range, kl_range, kl_range).to(box.dtype)
    
#     # Remove the origin (0,0,0)
#     all_hkl = all_hkl[torch.norm(all_hkl, dim=1) > 0.0]
    
#     # Convert h,k,l indices to reciprocal space vectors
#     kvectors = torch.matmul(all_hkl, reciprocal_box)
    
#     # Apply spherical cutoff based on k^2
#     k_squared = torch.einsum('ij,ij->i', kvectors, kvectors)
    
#     # Compute symmetry factors - only need to double for h>0
#     h = all_hkl[:, 0]
#     sym_factors = torch.ones_like(h)
#     sym_factors[h > 0] = 2.0
    
#     # Precalculating gaussian factors
#     gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / (alpha * alpha)) / k_squared

#     # Calculating all structure factors
#     k_dot_r = torch.matmul(kvectors, coords.T)
#     cos_k_dot_r = torch.cos(2 * torch.pi * k_dot_r)
#     sin_k_dot_r = torch.sin(2 * torch.pi * k_dot_r)

#     F_l_real = q.expand(kvectors.size(0), -1) - torch.einsum('kj,nij,ki->kn', kvectors, t, kvectors) * (2 * torch.pi) * (2 * torch.pi) / 3
#     F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
    
#     exp_k_dot_r = torch.complex(cos_k_dot_r, sin_k_dot_r)
#     exp_minus_k_dot_r = torch.complex(cos_k_dot_r, -sin_k_dot_r)
#     F_2 = torch.complex(F_l_real, F_l_imag)
#     structure_factors = torch.sum(F_2 * exp_k_dot_r, dim=1)
    
#     # Apply symmetry factors to each k-vector contribution
#     sym_factors = sym_factors.unsqueeze(1)
#     phi_expanded = (gaussian_factors.unsqueeze(1) * structure_factors.unsqueeze(1) * sym_factors) * exp_minus_k_dot_r
    
#     potential = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V)
#     field = 2 * (
#         torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
#     ) / V
#     k_outer = torch.vmap(torch.outer)(kvectors, kvectors).reshape(-1, 9)
#     field_grad = 4 * torch.pi * (
#         torch.matmul(phi_expanded.T, torch.complex(k_outer, torch.zeros_like(k_outer))).real.reshape(-1, 3, 3)
#     ) / V

#     # Now add in the self contributions to potential, field, and field gradient
#     alpha_over_root_pi = alpha / torch.sqrt(torch.tensor(torch.pi))
#     potential = potential - 2 * alpha_over_root_pi * q
#     field = field + alpha_over_root_pi * (4 * alpha * alpha / 3) * p
#     field_grad = field_grad + alpha_over_root_pi * (16 * alpha * alpha * alpha * alpha / 5) * t / 3

#     return potential, field, field_grad

@timeit
def long_range_potential(coords: torch.Tensor, q: torch.Tensor, p: torch.Tensor, 
                           t: torch.Tensor, box: torch.Tensor, alpha: float, max_hkl: int):
    """
    Parameters
    ----------
    coords: torch.Tensor
        Atomic coordinates (N, 3)
    q: torch.Tensor
        Charges (N,)
    p: torch.Tensor
        Dipoles (N, 3)
    t: torch.Tensor
        Quadrupoles (N, 3, 3)
    box: torch.Tensor
        Box vectors (3, 3)
    alpha: torch.Tensor
        Ewald splitting parameter
    max_hkl: int
        Maximum h,k,l index values for reciprocal space sum
    
    Returns
    -------
    potential: torch.Tensor
        Electric potential at each atom (N,)
    field: torch.Tensor
        Electric field at each atom (N, 3)
    field_grad: torch.Tensor
        Electric field gradient at each atom (N, 3, 3)
    """
    # Reciprocal lattice vectors
    V = torch.det(box)  # volume of box
    reciprocal_box = torch.stack((
        torch.linalg.cross(box[1], box[2]),
        torch.linalg.cross(box[2], box[0]),
        torch.linalg.cross(box[0], box[1])
    )) / V

    # We optimize for h ≥ 0, but keep full range for k and l
    h_range = torch.arange(0, max_hkl + 1, device=coords.device)
    kl_range = torch.arange(-max_hkl, max_hkl + 1, device=coords.device)
    
    # Create all combinations and convert to float64
    all_hkl = torch.cartesian_prod(h_range, kl_range, kl_range).to(box.dtype)
    
    # Remove the origin (0,0,0)
    all_hkl = all_hkl[torch.norm(all_hkl, dim=1) > 0.0]
    
    # Convert h,k,l indices to reciprocal space vectors
    kvectors = torch.matmul(all_hkl, reciprocal_box)
    
    # Apply spherical cutoff based on k^2
    k_squared = torch.einsum('ij,ij->i', kvectors, kvectors)
    # k_squared = torch.sum(kvectors**2, dim=1)
    
    # Compute symmetry factors - only need to double for h>0
    h = all_hkl[:, 0]
    sym_factors = torch.ones_like(h)
    sym_factors[h > 0] = 2.0
    
    # Precalculating gaussian factors
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
    
    # Apply symmetry factors to each k-vector contribution
    sym_factors = sym_factors.unsqueeze(1)
    phi_expanded = (gaussian_factors.unsqueeze(1) * structure_factors.unsqueeze(1) * sym_factors) * exp_minus_k_dot_r
    
    potential = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V)
    field = 2 * (
        torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
    ) / V
    # k_outer = torch.vmap(torch.outer)(kvectors, kvectors).reshape(-1, 9)
    k_outer = torch.einsum('bi,bj->bij', kvectors, kvectors).view(-1, 9)
    field_grad = 4 * torch.pi * (
        torch.matmul(phi_expanded.T, torch.complex(k_outer, torch.zeros_like(k_outer))).real.view(-1, 3, 3)
    ) / V

    # Now add in the self contributions to potential, field, and field gradient
    alpha_over_root_pi = alpha / torch.sqrt(torch.tensor(torch.pi))
    potential = potential - 2 * alpha_over_root_pi * q
    field = field + alpha_over_root_pi * (4 * alpha * alpha / 3) * p
    field_grad = field_grad + alpha_over_root_pi * (16 * alpha * alpha * alpha * alpha / 5) * t / 3

    return potential, field, field_grad


@torch.compile
class Ewald(nn.Module):
    def __init__(self, alpha: float, max_hkl: int):
        super().__init__()
        self.alpha = alpha
        # We optimize for h ≥ 0, but keep full range for k and l
        h_range = torch.arange(0, max_hkl + 1, device=coords.device)
        kl_range = torch.arange(-max_hkl, max_hkl + 1, device=coords.device)
        
        # Create all combinations and convert to float64
        all_hkl = torch.cartesian_prod(h_range, kl_range, kl_range).to(box.dtype)
        
        # Remove the origin (0,0,0)
        all_hkl = all_hkl[torch.norm(all_hkl, dim=1) > 0.0]
        h = all_hkl[:, 0]
        sym_factors = torch.ones_like(h)
        sym_factors[h > 0] = 2.0

        self.sym_factors = sym_factors.unsqueeze(1)
        self.all_hkl = all_hkl
        self.alpha2 = alpha * alpha
        self.alpha4 = alpha ** 4
    
    def forward(self, coords, q, p, t, box, boxInv, V):
        kvectors = torch.matmul(self.all_hkl, boxInv)
    
        # Apply spherical cutoff based on k^2
        k_squared = torch.einsum('ij,ij->i', kvectors, kvectors)
        # k_squared = torch.sum(kvectors**2, dim=1)
        
        # Precalculating gaussian factors
        gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / self.alpha2) / k_squared

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
        
        # Apply symmetry factors to each k-vector contribution
        phi_expanded = (gaussian_factors.unsqueeze(1) * structure_factors.unsqueeze(1) * self.sym_factors) * exp_minus_k_dot_r
        
        potential = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V)
        field = 2 * (
            torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
        ) / V
        # k_outer = torch.vmap(torch.outer)(kvectors, kvectors).reshape(-1, 9)
        k_outer = torch.einsum('bi,bj->bij', kvectors, kvectors).view(-1, 9)
        field_grad = 4 * torch.pi * (
            torch.matmul(phi_expanded.T, torch.complex(k_outer, torch.zeros_like(k_outer))).real.view(-1, 3, 3)
        ) / V

        # Now add in the self contributions to potential, field, and field gradient
        alpha_over_root_pi = self.alpha / torch.sqrt(torch.tensor(torch.pi))
        potential = potential - 2 * alpha_over_root_pi * q
        field = field + alpha_over_root_pi * (4 * self.alpha2 / 3) * p
        field_grad = field_grad + alpha_over_root_pi * (16 * self.alpha4 / 5) * t / 3

        return potential, field, field_grad




if __name__ == '__main__':
    device = 'cuda'
    dtype = torch.float64
    num_water = 216
    q = torch.tensor([-0.39, 0.195, 0.195] * num_water, device=device, dtype=dtype)
    p = torch.rand((num_water*3, 3), dtype=dtype, device=device)
    t = np.random.rand(num_water*3, 3, 3)
    t = t + np.einsum('kij->kji', t)
    t[:, [0, 1, 2], [0, 1, 2]] -= np.sum(t[:, [0, 1, 2], [0, 1, 2]],axis=1,keepdims=True) / 3
    t = torch.tensor(t, device=device, dtype=dtype)
    coords = torch.rand((num_water*3, 3), dtype=dtype, device=device, requires_grad=True)
    box = torch.eye(3, device=device, dtype=dtype)
    boxV = torch.det(box)
    boxInv = torch.inverse(box)

    long_range_potential(coords, q, p, t, box, 0.14, 6)

    # lr = timeit(Ewald(0.14, 6))
    # lr(coords, q, p, t, box, boxInv, boxV)
