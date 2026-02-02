import torch
import torch.nn as nn
from typing import Optional
from .pme import compute_pme
try:
    import torchff
    import torchff_pme
except ImportError:
    pass

class PME(nn.Module):
    def __init__(self, alpha: float, max_hkl: int, rank: int, use_customized_ops: bool = False):
        super().__init__()
        self.alpha = float(alpha)
        self.max_hkl = int(max_hkl)
        self.rank = int(rank)
        self.use_customized_ops = use_customized_ops

    def _pack_quadrupoles(self, t: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 6 unique components from an (N, 3, 3) quadrupole tensor
        into an (N, 6) tensor: [xx, xy, xz, yy, yz, zz].
        
        Using torch.stack ensures gradients are tracked back to the original 3x3 tensor.
        """
        # If t is already (N, 6), assume it is packed and return it
        if t.shape[-1] == 6 and t.ndim == 2:
            return t
        
        # If t is (N, 3, 3), extract unique components
        if t.ndim == 3 and t.shape[-1] == 3:
            # We assume t is symmetric. We extract the upper triangle.
            # xx, xy, xz, yy, yz, zz
            return torch.stack([
                t[:, 0, 0],  # Qxx
                t[:, 0, 1],  # Qxy
                t[:, 0, 2],  # Qxz
                t[:, 1, 1],  # Qyy
                t[:, 1, 2],  # Qyz
                t[:, 2, 2]   # Qzz
            ], dim=1)
            
        # If t is flattened (N, 9), extract indices
        if t.ndim == 2 and t.shape[-1] == 9:
             return torch.stack([
                t[:, 0], t[:, 1], t[:, 2],
                t[:, 4], t[:, 5], t[:, 8]
             ], dim=1)
             
        raise ValueError(f"Quadrupole tensor t has unexpected shape: {t.shape}")

    def forward(self, coords: torch.Tensor, box: torch.Tensor, q: torch.Tensor, p: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
        # 1. Pre-process Quadrupoles for both C++ and Python paths
        t_packed = t
        if self.rank >= 2 and t is not None:
            t_packed = self._pack_quadrupoles(t)
        if self.use_customized_ops:
            return self._forward_cpp(coords, box, q, p, t_packed)
        else:
            return self._forward_python(coords, box, q, p, t_packed)

    def _forward_cpp(self, coords, box, q, p, t_packed):
        # Returns: (phi, E, EG, energy, forces)
        # We pass the packed (N, 6) tensor here.
        # The C++ extension will return gradients for t_packed (N, 6).
        # PyTorch will automatically map those gradients back to the original (N, 3, 3) t input.
        return torch.ops.torchff.pme_long_range(
            coords, box, q, p, t_packed,
            self.max_hkl, self.rank, self.alpha
        )

    def _forward_python(self, coords, box, q, p, t_packed):
        # 1. Compute PME terms
        ret = compute_pme(coords, box, q, p, t_packed, self.alpha, self.max_hkl, self.rank)
        
        # 2. Unpack results based on rank
        pot = ret if self.rank == 0 else ret[0]
        field = ret[1] if self.rank >= 1 else torch.zeros_like(p)
        EG = ret[2] if self.rank >= 2 else (torch.zeros_like(t_packed) if t_packed is not None else None)
        # 3. Calculate Energy Terms
        term_q = 0.5 * torch.sum(q * pot)
        term_p = 0.0
        if self.rank >= 1 and p is not None:
            term_p = -0.5 * torch.sum(p * field)
        term_t = 0.0
        if self.rank >= 2 and t_packed is not None:
            # Extract diagonal and off-diagonal elements from the (N,3,3) field gradient EG
            eg_xx = EG[:, 0, 0]
            eg_xy = EG[:, 0, 1]
            eg_xz = EG[:, 0, 2]
            eg_yy = EG[:, 1, 1]
            eg_yz = EG[:, 1, 2]
            eg_zz = EG[:, 2, 2]

            # Calculate contraction Q : grad E
            # t_packed indices correspond to: 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz
            contraction = (
                t_packed[:, 0] * eg_xx +
                t_packed[:, 3] * eg_yy +
                t_packed[:, 5] * eg_zz +
                2.0 * (t_packed[:, 1] * eg_xy + t_packed[:, 2] * eg_xz + t_packed[:, 4] * eg_yz)
            )

            term_t = -(1.0/2.0) * torch.sum(contraction)
        energy = term_q + term_p + term_t
        return pot, field, EG, energy
