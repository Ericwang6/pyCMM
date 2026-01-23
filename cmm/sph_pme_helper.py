import torch
import torch.nn as nn
from typing import Optional
from .sph_pme import compute_pme
from .multipole import (
        computeSphericalDipoles
)

try:
    import torchff
    import torchff_pme
except ImportError:
    pass
import torch
import math
def cartesian_to_spherical_quadrupoles(quad_cart: torch.Tensor):
    """
    Converts standard Cartesian Quadrupoles to Real Spherical Harmonic coefficients.

    Input:
      quad_cart: (N, 6) tensor of [Qxx, Qxy, Qxz, Qyy, Qyz, Qzz]
      Assumes Trace(Q) = 0.

    Output:
      quad_sph: (N, 5) tensor of [Q20, Q21c, Q21s, Q22c, Q22s]
      Matches the basis functions used in standard PME implementations.
    """
    # 1. Unpack Raw Components (No scaling applied yet)
    q_xx = quad_cart[:, 0]
    q_xy = quad_cart[:, 1]
    q_xz = quad_cart[:, 2]
    q_yy = quad_cart[:, 3]
    q_yz = quad_cart[:, 4]
    q_zz = quad_cart[:, 5]

    # 2. Define Conversion Constants
    # These derived factors ensure: Sum(Q_cart * Operator_cart) == Sum(Q_sph * Operator_sph)
    SQRT3 = torch.sqrt(torch.tensor(3.0))

    # 3. Compute Spherical Coefficients
    # Q2,0 (Axial): Associates with (3z^2 - r^2)/2
    # In standard convention (Stone), Q20 = Qzz.
    # However, because of the specific PME spreading code you showed (which divides by 3),
    # and the basis definition, we often pass 3*Qzz so that (3*Qzz)/3 = Qzz acts on the basis.
    # We will stick to the PHYSICAL definition here:
    q20 = q_zz

    # Q2,1c (Tilt XZ): Associates with sqrt(3)xz
    # Factor 2/sqrt(3) accounts for the '2' in (xz+zx) and normalizing the basis.
    q21c = (2.0 / SQRT3) * q_xz

    # Q2,1s (Tilt YZ): Associates with sqrt(3)yz
    q21s = (2.0 / SQRT3) * q_yz

    # Q2,2c (Planar Diagonal): Associates with sqrt(3)/2 * (x^2 - y^2)
    # Factor matches the difference in diagonals.
    q22c = (1.0 / SQRT3) * (q_xx - q_yy)

    # Q2,2s (Planar Off-Diagonal): Associates with sqrt(3)xy
    q22s = (2.0 / SQRT3) * q_xy
    #Stacj 
    quad_s = torch.stack((q20, q21c, q21s, q22c, q22s), dim=1)
    #scale and return
    return quad_s * 1.5

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
        else:
            p = computeSphericalDipoles(p) 
        if t is None: 
            # Default to Cartesian 3x3 input shape
            t = torch.zeros((N, 3, 3), device=device, dtype=dtype)
        else:
            t = cartesian_to_spherical_quadrupoles(t)

    return q,p,t

class PME_Spherical(nn.Module):
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
            #DEBUGGING E_K
            E_k = ret[3] if len(ret) > 3 else 0
        else:
            phi = ret
            E = torch.zeros_like(p)
            EG = torch.zeros_like(t)

        # 2. Compute Scalar Energy Manually
        # Formula for Spherical Harmonics contraction
        # U = 0.5 * (q*phi) - 0.5 * (p*E) - 0.5 * (t*EG)
        term_q = 0.5 * torch.sum(q * phi)
        term_p = -0.5 * torch.sum(p * E)
        term_t = -0.5 * torch.sum(t * EG)
        print(f"SPHERICAL PME MONOPOLE   ENERGY: {term_q}")
        print(f"SPHERICAL PME DIPOLE     ENERGY: {term_p}")
        print(f"SPHERICAL PME QUADRUPOLE ENERGY: {term_t}")
        energy = term_q + term_p + term_t
        forces = None 

        # Return 5 items to match C++ signature
        return phi, E, EG, energy, E_k, forces
