import torch
import torch.nn as nn
from typing import Optional
from .pme import compute_pme
from .multipole import (
    convertMultipolesToPolytensor,
    createPMEPolytensor, 
    split_spherical_polytensor
)

try:
    import torchff
    import torchff_pme
except ImportError:
    pass

def get_pme_multipoles(q: torch.Tensor, p: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
    """
    Converts Cartesian multipoles to Spherical Harmonic basis.
    Wraps in no_grad to prevent 'IndexPutBackward' errors if gradients 
    w.r.t multipoles are not strictly required through the conversion.
    """
    # Use no_grad to fix the Autograd/Dynamo warnings you saw earlier
    with torch.no_grad():
        device = q.device
        dtype = q.dtype
        N = q.shape[0]

        if p is None: 
            p = torch.zeros((N, 3), device=device, dtype=dtype)
        if t is None: 
            # Default to Cartesian 3x3 input shape
            t = torch.zeros((N, 3, 3), device=device, dtype=dtype)

        # Cartesian -> Polytensor -> Spherical PME Basis
        multipoles = convertMultipolesToPolytensor(q, p, t)
        multipoles_s = createPMEPolytensor(multipoles)
        
        # Split back into (q, p, t) where t is now shape (N, 5)
        q_s, p_s, t_s = split_spherical_polytensor(multipoles_s)
        
    return q_s, p_s, t_s

class PME(nn.Module):
    def __init__(self, alpha: float, max_hkl: int, rank: int, use_customized_ops: bool = False):
        super().__init__()
        self.alpha = float(alpha)
        self.max_hkl = int(max_hkl)
        self.rank = int(rank)
        self.use_customized_ops = use_customized_ops

    def forward(self, coords: torch.Tensor, box: torch.Tensor, q: torch.Tensor, p: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
        q_s, p_s, t_s = get_pme_multipoles(q, p, t)

        # 2. Dispatch
        if self.use_customized_ops:
            return self._forward_cpp(coords, box, q_s, p_s, t_s)
        else:
            return self._forward_python(coords, box, q_s, p_s, t_s)

    def _forward_cpp(self, coords, box, q, p, t):
        # Returns: (phi, E, EG, energy, forces)
        return torch.ops.torchff.pme_long_range(
            coords, box, q, p, t, 
            self.max_hkl, self.rank, self.alpha
        )

    def _forward_python(self, coords, box, q, p, t):
        # 1. Compute Potentials (Returns: phi, E, EG)
        ret = compute_pme(coords, box, q, p, t, self.alpha, self.max_hkl, self.rank)
        
        if isinstance(ret, tuple):
            phi = ret[0]
            # Handle cases where rank < 2 might return fewer items
            E   = ret[1] if len(ret) > 1 else torch.zeros_like(p)
            EG  = ret[2] if len(ret) > 2 else torch.zeros_like(t)
        else:
            phi = ret
            E = torch.zeros_like(p)
            EG = torch.zeros_like(t)

        # 2. Compute Scalar Energy Manually
        # Formula for Spherical Harmonics contraction
        # U = 0.5 * (q*phi) - 0.5 * (p*E) - 1/6 * (t*EG)
        term_q = 0.5 * torch.sum(q * phi)
        term_p = -0.5 * torch.sum(p * E)
        term_t = -(1.0/6.0) * torch.sum(t * EG)
        energy = term_q + term_p + term_t
        forces = None 

        # Return 5 items to match C++ signature
        return phi, E, EG, energy, forces
