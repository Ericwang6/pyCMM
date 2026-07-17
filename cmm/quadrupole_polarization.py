"""Direct (non-self-consistent) quadrupole polarization.

Each atom carries an effective isotropic quadrupole polarizability C
(ffxml parameter `quad_pol`, a.u.). The induced quadrupole responds to the
*permanent* electric field gradient only — it is not fed back into the
polarization SCF, so the term is a cheap per-atom contraction:

    U_qpol,i = -1/2 C_i |G0_i|^2 ,   G0 = traceless(grad E_perm)

with |G0|^2 the full double contraction (off-diagonal components counted
twice). Any convention factor in Theta = C grad E is absorbed into the fitted
C. For an ion at distance r from a bare point charge q the traceless
|grad E|^2 = 6 q^2 / r^6, so U_qpol = -3 C q^2 / r^6.

Limitations: because the response is to the permanent gradient only, mutual
(induced-dipole -> gradient) couplings are neglected. Forces are conservative:
the energy is an explicit function of coordinates, handled by autograd.
"""
import torch

# packed second-derivative order used throughout the code base
# (see electrostatics.computePermanentElectricPotentialExpansion)
_PACKED = ('xx', 'xy', 'xz', 'yy', 'yz', 'zz')
_DIAG = (0, 3, 5)
_OFFDIAG = (1, 2, 4)


def quadrupole_polarization_energy(efield_grad_packed: torch.Tensor,
                                   quad_pol: torch.Tensor) -> torch.Tensor:
    """Per-atom direct quadrupole polarization energy, shape (N,).

    efield_grad_packed: (N, 6) packed [xx, xy, xz, yy, yz, zz] second
        derivatives of the permanent potential (sign convention irrelevant).
    quad_pol: (N,) effective isotropic quadrupole polarizabilities C (a.u.).
    """
    g = efield_grad_packed
    trace_third = (g[:, 0] + g[:, 3] + g[:, 5]) / 3.0
    d = torch.stack([g[:, i] - trace_third for i in _DIAG], dim=1)
    o = g[:, _OFFDIAG]
    normsq = torch.sum(d * d, dim=1) + 2.0 * torch.sum(o * o, dim=1)
    return -0.5 * quad_pol * normsq
