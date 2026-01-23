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

    def forward(self, coords: torch.Tensor, box: torch.Tensor, q: torch.Tensor, p: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
        if self.use_customized_ops:
            return self._forward_cpp(coords, box, q, p, t)
        else:
            return self._forward_python(coords, box, q, p, t)

    def _forward_cpp(self, coords, box, q, p, t):
        # Returns: (phi, E, EG, energy, forces)
        return torch.ops.torchff.pme_long_range(
            coords, box, q, p, t, 
            self.max_hkl, self.rank, self.alpha
        )
    def _forward_python(self, coords, box, q, p, t):
        # 1. Compute PME terms
        ret = compute_pme(coords, box, q, p, t, self.alpha, self.max_hkl, self.rank)
        # 2. Unpack results based on rank
        pot = ret if self.rank == 0 else ret[0]
        field = ret[1] if self.rank >= 1 else torch.zeros_like(p)
        EG = ret[2] if self.rank >= 2 else torch.zeros_like(t)
        #DEBUGGING E_K
        #E_k = ret[3] if self.rank>=2 else 0
        # 3. Calculate Energy Terms
        # Term Q (Charge Energy): 0.5 * sum(q * phi)
        term_q = 0.5 * torch.sum(q * pot)
        # Term P (Dipole Energy): -0.5 * sum(p * E)
        term_p = 0.0
        if self.rank >= 1:
            term_p = -0.5 * torch.sum(p * field)
        # Term T (Quadrupole Energy): -(1/2) * sum(Q : gradE)
        term_t = 0.0
        if self.rank >= 2:
            #term_t = -(1.0/2.0) * torch.sum(t * EG)
            eg_xx = EG[:, 0, 0]
            eg_xy = EG[:, 0, 1]
            eg_xz = EG[:, 0, 2]
            eg_yy = EG[:, 1, 1]
            eg_yz = EG[:, 1, 2]
            eg_zz = EG[:, 2, 2]

            # 2. Calculate the dot product Q : sum(grad E)
            contraction = (
                t[:, 0] * eg_xx +
                t[:, 3] * eg_yy +
                t[:, 5] * eg_zz +
                2.0 * (t[:, 1] * eg_xy + t[:, 2] * eg_xz + t[:, 4] * eg_yz)
            )

            term_t = -(1.0/2.0) * torch.sum(contraction)
        # Total Reciprocal Energy
        print(f"CARTESIAN PME MONOPOLE   RECIPROCAL ENERGY: {term_q}")
        print(f"CARTESIAN PME DIPOLE     RECIPROCAL ENERGY: {term_p}")
        print(f"CARTESIAN PME QUADRUPOLE RECIPROCAL ENERGY: {term_t}")
        energy = term_q + term_p + term_t
        forces = None
        return pot, field, EG, energy, forces

