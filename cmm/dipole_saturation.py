"""Dipole saturation as a nonlinear self-energy in the polarization functional.

The dipole self-energy of atom i is augmented with a saturation potential

    U_self,i = 1/2 mu_i^T alpha_i^-1 mu_i + U_sat,i(mu_i)

    U_sat,i(mu_i) = U_iso(m) + U_ani(mu_i . n_i),   m = |mu_i|

where n_i is the unit permanent-field axis at atom i. With the reduced
magnitudes u = m / m0 and v = (mu . n) / m0, m0 = alpha_iso * field_scale,

    U_iso(m)      = 1/2 c_iso (m0^2/alpha) [u^2 - ln(1 + u^2)]
    U_ani(mu . n) = -c_ani (m0^2/alpha) v^4 exp(-v^2/w^2)

Both vanish *with vanishing curvature* as mu -> 0 (the v^4 form, rather than
the naive v^2 bump, is what makes U_ani''(0) = 0), so the free-ion
polarizability is recovered exactly at zero field. The dipole stationarity
condition of the augmented functional is

    alpha_i^-1 mu_i + grad U_sat,i(mu_i) = E_i^tot .            (*)

Because grad U_sat(mu) = K(mu) mu with the symmetric "secant" matrix

    K(mu) = (U_iso'(m)/m) I + (U_ani'(s)/s) n n^T,   s = mu . n,

equation (*) is solved by a Picard sequence of *linear* solves in which the
dipole self-block alpha^-1 is replaced by B_sec = alpha^-1 + K(mu). The
converged mu satisfies (*) exactly, so forces are conservative and the
polarization energy is the quadratic functional (with the original alpha^-1)
plus sum_i U_sat,i(mu_i).

The differential response is governed by the Hessian

    B = alpha^-1 + (U_iso'(m)/m)(I - mu mu^T/m^2) + U_iso''(m) mu mu^T/m^2
              + U_ani''(s) n n^T

whose perpendicular component damps monotonically (U_iso'/m >= 0) while the
component along n first softens (U_ani''(0) < 0: parallel polarizability
rises) and then hardens at larger dipole, matching the QM pair-polarizability
anisotropy.

All quantities are in atomic units.
"""
import torch

# Default field scale defining the reduced dipole u = m / (alpha_iso * E0).
# 0.05 a.u. ~ 257 MV/cm. Every function below also accepts a per-atom tensor
# for field_scale (the ffxml parameter `sat_e0`), which decouples the
# saturation *onset* (E0) from its *depth* (c_iso).
SAT_FIELD_SCALE = 0.05

# Smallest allowed fraction of alpha^-1 left in the parallel dipole block of
# the secant matrix; bounds the parallel polarizability enhancement and keeps
# the linear solves well-conditioned.
SAT_SPD_FLOOR = 0.1

_EPS = 1e-30


def unit_field_axis(efield: torch.Tensor, tol: float = 1e-10) -> torch.Tensor:
    """Zero-safe unit vector along the (permanent) electric field.

    Atoms with |E| < tol get a zero axis, which switches the anisotropic
    saturation term off for them.
    """
    norm = torch.norm(efield, dim=-1, keepdim=True)
    return torch.where(norm > tol, efield / norm.clamp_min(_EPS), torch.zeros_like(efield))


def _reduced(mu, alpha_iso, n_axis, w, field_scale):
    alpha_iso = alpha_iso.clamp_min(_EPS)
    m0 = alpha_iso * field_scale
    u2 = torch.sum(mu * mu, dim=-1) / (m0 * m0)
    s = torch.sum(mu * n_axis, dim=-1)
    v = s / m0
    w = w.clamp_min(1e-6)
    return alpha_iso, m0, u2, s, v, w


def saturation_energy(mu, alpha_iso, n_axis, c_iso, c_ani, w, field_scale=SAT_FIELD_SCALE):
    """Per-atom saturation self-energy U_sat,i(mu_i), shape (N,).

    mu:        (N, 3) induced dipoles
    alpha_iso: (N,)   isotropic free-ion polarizability (trace/3)
    n_axis:    (N, 3) unit permanent-field axis (zero-safe, see unit_field_axis)
    c_iso, c_ani, w: (N,) per-atom saturation parameters
    """
    alpha_iso, m0, u2, s, v, w = _reduced(mu, alpha_iso, n_axis, w, field_scale)
    pref = m0 * m0 / alpha_iso
    e_iso = 0.5 * c_iso * pref * (u2 - torch.log1p(u2))
    vw2 = (v / w) ** 2
    e_ani = -c_ani * pref * v ** 4 * torch.exp(-vw2)
    return e_iso + e_ani


def saturation_gradient(mu, alpha_iso, n_axis, c_iso, c_ani, w, field_scale=SAT_FIELD_SCALE):
    """grad_mu U_sat, shape (N, 3). Uses the exact (unclamped) derivatives."""
    k_iso, k_ani = saturation_secant_coeffs(mu, alpha_iso, n_axis, c_iso, c_ani, w,
                                            field_scale, clamp_spd=False)
    s = torch.sum(mu * n_axis, dim=-1)
    return k_iso.unsqueeze(-1) * mu + (k_ani * s).unsqueeze(-1) * n_axis


def saturation_secant_coeffs(mu, alpha_iso, n_axis, c_iso, c_ani, w,
                             field_scale=SAT_FIELD_SCALE, clamp_spd=True):
    """Coefficients (k_iso, k_ani), each (N,), of the secant matrix

        K(mu) = k_iso I + k_ani n n^T   with   grad U_sat(mu) = K(mu) mu .

    k_iso = U_iso'(m)/m = (c_iso/alpha) u^2/(1+u^2)                (>= 0)
    k_ani = U_ani'(s)/s = -(c_ani/alpha) (4 v^2 - 2 v^4/w^2) e^{-v^2/w^2}

    With clamp_spd, k_ani is floored so the parallel eigenvalue of
    alpha^-1 + K stays >= SAT_SPD_FLOOR / alpha.
    """
    alpha_iso, m0, u2, s, v, w = _reduced(mu, alpha_iso, n_axis, w, field_scale)
    k_iso = (c_iso / alpha_iso) * u2 / (1.0 + u2)
    v2 = v * v
    vw2 = v2 / (w * w)
    k_ani = -(c_ani / alpha_iso) * (4.0 * v2 - 2.0 * v2 * vw2) * torch.exp(-vw2)
    if clamp_spd:
        k_ani_floor = (SAT_SPD_FLOOR - 1.0) / alpha_iso - k_iso
        k_ani = torch.maximum(k_ani, k_ani_floor)
    return k_iso, k_ani


def saturation_secant_matrix(mu, alpha_iso, n_axis, c_iso, c_ani, w,
                             field_scale=SAT_FIELD_SCALE):
    """K(mu) as (N, 3, 3), to be added to alpha^-1 in the dipole self-block."""
    k_iso, k_ani = saturation_secant_coeffs(mu, alpha_iso, n_axis, c_iso, c_ani, w, field_scale)
    eye = torch.eye(3, device=mu.device, dtype=mu.dtype).expand(mu.shape[0], 3, 3)
    nnT = n_axis.unsqueeze(-1) * n_axis.unsqueeze(-2)
    return k_iso.view(-1, 1, 1) * eye + k_ani.view(-1, 1, 1) * nnT


def saturation_hessian(mu, alpha_iso, n_axis, c_iso, c_ani, w, field_scale=SAT_FIELD_SCALE):
    """Hessian grad^2_mu U_sat, shape (N, 3, 3) (exact, unclamped).

    Governs the differential response: added perpendicular stiffness
    U_iso'(m)/m, added parallel stiffness U_iso''(m), and the sign-changing
    axial term U_ani''(s) along n.
    """
    alpha_iso, m0, u2, s, v, w = _reduced(mu, alpha_iso, n_axis, w, field_scale)
    one_p = 1.0 + u2

    k_perp = (c_iso / alpha_iso) * u2 / one_p
    k_par = (c_iso / alpha_iso) * u2 * (3.0 + u2) / (one_p * one_p)

    m = torch.norm(mu, dim=-1, keepdim=True)
    mu_hat = torch.where(m > 1e-14, mu / m.clamp_min(_EPS), torch.zeros_like(mu))
    eye = torch.eye(3, device=mu.device, dtype=mu.dtype).expand(mu.shape[0], 3, 3)
    mmT = mu_hat.unsqueeze(-1) * mu_hat.unsqueeze(-2)
    nnT = n_axis.unsqueeze(-1) * n_axis.unsqueeze(-2)

    v2 = v * v
    vw2 = v2 / (w * w)
    h_ani = -(c_ani / alpha_iso) * (12.0 * v2 - 18.0 * v2 * vw2 + 4.0 * v2 * vw2 * vw2) * torch.exp(-vw2)

    return (k_perp.view(-1, 1, 1) * (eye - mmT)
            + k_par.view(-1, 1, 1) * mmT
            + h_ani.view(-1, 1, 1) * nnT)
