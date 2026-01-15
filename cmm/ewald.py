import torch
import torch.nn as nn
from typing import Optional

try:
    import torchff
    import torchff_ewald
except Exception as e:
    pass


class EwaldFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, box, q, p, t, max_hkl, rank, alpha):
        #Run CUDA kernel
        pot, fld, grad, energy, forces = torch.ops.torchff.ewald_long_range(
            coords, box, q, p, t, max_hkl, rank, alpha
        )

        # Save the Field and Field Gradient for the backward pass
        ctx.save_for_backward(fld, grad, forces)
        ctx.rank = rank

        return pot, fld, grad, energy, forces

    @staticmethod
    def backward(ctx, grad_pot, grad_fld, grad_grad, grad_energy, grad_forces):
        fld, grad_field_tensor, forces = ctx.saved_tensors
        rank = ctx.rank

        # --- CALCULATE GRADIENTS ---
        # 1. Gradient w.r.t Position (The Translational Force)
        d_coords = -forces * grad_energy

        # 2. Gradient w.r.t Multipoles (The Torque Source)
        d_p = None
        if rank >= 1:
            d_p = -fld * grad_energy

        # 3. Gradient w.r.t Quadrupoles
        d_t = None
        if rank >= 2:
            d_t = -(1.0/3.0) * grad_field_tensor * grad_energy

        return d_coords, None, None, d_p, d_t, None, None, None

class Ewald(nn.Module):
    def __init__(self, alpha: float, max_hkl: int, rank: int, use_customized_ops: bool = False):
        super().__init__()
        sym_factors = []
        all_hkl = []
        for h in range(0, max_hkl+1):
            for k in range(-max_hkl, max_hkl+1):
                for l in range(-max_hkl, max_hkl+1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    all_hkl.append([float(h), float(k), float(l)])
                    sym_factors.append(2.0 if h > 0 else 1.0)
                        
        self.register_buffer('sym_factors', torch.tensor(sym_factors).unsqueeze(1))
        self.register_buffer('all_hkl', torch.tensor(all_hkl))
        self.max_hkl = max_hkl
        self.alpha = alpha
        self.alpha2 = alpha * alpha
        self.alpha_over_root_pi = self.alpha / torch.sqrt(torch.tensor(torch.pi))
        self.rank = rank
        self.use_customized_ops = use_customized_ops
    
    def forward(self, coords: torch.Tensor, box: torch.Tensor, q: torch.Tensor, p: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
        if self.use_customized_ops:
            return self._forward_cpp(coords, box, q, p, t)
        else:
            return self._forward_python(coords, box, q, p, t)
    
    def _forward_cpp(self, coords, box, q, p, t):
        dev = coords.device
        dtype = torch.float64
        coords = coords.to(device=dev, dtype=dtype).contiguous()
        box    = box.to(device=dev, dtype=dtype).contiguous()
        q      = q.to(device=dev, dtype=dtype).contiguous()
        N = q.shape[0]
        if p is None:
            p = torch.zeros((N, 3), device=dev, dtype=dtype)
        else:
            p = p.to(device=dev, dtype=dtype).contiguous()

        if t is None:
            t = torch.zeros((N, 3, 3), device=dev, dtype=dtype)
        else:
            t = t.to(device=dev, dtype=dtype).contiguous()
        return EwaldFunction.apply(
            coords, box, q, p, t, self.max_hkl, self.rank, float(self.alpha)
        )
        return pot, fld, grad, energy, forces
    def _forward_python(self, coords: torch.Tensor, box: torch.Tensor, q: torch.Tensor, p: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
        box_inv = torch.inverse(box)
        V = torch.det(box)

        # Convert h,k,l indices to reciprocal space vectors
        kvectors = torch.matmul(self.all_hkl, box_inv)
    
        # Apply spherical cutoff based on k^2
        k_squared = torch.einsum('ij,ij->i', kvectors, kvectors)
        gaussian_factors = torch.exp(-torch.pi * torch.pi * k_squared / self.alpha2) / k_squared

        # Calculating all structure factors
        k_dot_r = torch.matmul(kvectors, coords.T)
        cos_k_dot_r = torch.cos(2 * torch.pi * k_dot_r)
        sin_k_dot_r = torch.sin(2 * torch.pi * k_dot_r)

        if self.rank == 2:
            F_l_real = q.expand(kvectors.size(0), -1) - torch.einsum('kj,nij,ki->kn', kvectors, t, kvectors) * (2 * torch.pi) * (2 * torch.pi) / 3
            F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
        elif self.rank == 1:
            F_l_real = q.expand(kvectors.size(0), -1)
            F_l_imag = torch.matmul(kvectors, p.T) * 2 * torch.pi
        else:
            F_l_real = q.expand(kvectors.size(0), -1)
            F_l_imag = torch.zeros(kvectors.size(0), q.size(0), device=q.device, dtype=q.dtype)
        
        exp_k_dot_r = torch.complex(cos_k_dot_r, sin_k_dot_r)
        exp_minus_k_dot_r = torch.complex(cos_k_dot_r, -sin_k_dot_r)
        F_2 = torch.complex(F_l_real, F_l_imag)
        structure_factors = torch.sum(F_2 * exp_k_dot_r, dim=1)
        # Apply symmetry factors to each k-vector contribution
        phi_expanded = (gaussian_factors.unsqueeze(1) * structure_factors.unsqueeze(1) * self.sym_factors) * exp_minus_k_dot_r
        
        potential = torch.sum(phi_expanded.real, dim=0) / (torch.pi * V)
        potential = potential - 2 * self.alpha_over_root_pi * q  # self contributions
        if self.rank == 0:
            return potential
        
        field = 2 * (
            torch.matmul(phi_expanded.T, torch.complex(torch.zeros_like(kvectors), kvectors)).real
        ) / V
        field = field + self.alpha_over_root_pi * (4 * self.alpha2 / 3) * p

        if self.rank == 1:
            return potential, field
        
        k_outer = torch.einsum('bi,bj->bij', kvectors, kvectors).reshape(-1, 9)
        field_grad = 4 * torch.pi * (
            torch.matmul(phi_expanded.T, torch.complex(k_outer, torch.zeros_like(k_outer))).real.reshape(-1, 3, 3)
        ) / V
        field_grad = field_grad + self.alpha_over_root_pi * (16 * self.alpha2 * self.alpha2 / 5) * t / 3

        if self.rank == 2:
            return potential, field, field_grad
