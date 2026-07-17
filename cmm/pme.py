import torch
from scipy import special
from torch.special import erfc
import math
import numpy as np
from itertools import combinations
torch.set_printoptions(profile="full")
if torch.cuda.is_available():
    torch.set_default_device("cuda")

########################################################################################################################
def get_recip_vectors(N,box):
    """
    Get recip lattice vectors of grid
    """
    N = torch.as_tensor(N, device=box.device, dtype=box.dtype)
    Nj_Aji_star = (N.reshape((1,3)) * torch.linalg.inv(box)).T
    return Nj_Aji_star

def get_u_reference(coords, Nj_Aji_star):
    """
    Maps particle positions to grid
    """
    bspline_order = 6
    R_in_m_basis = torch.einsum("ij,kj->ki",Nj_Aji_star,coords)
    m_u0 = torch.ceil(R_in_m_basis).to(torch.int64)
    u0 = (m_u0 - R_in_m_basis) + bspline_order/2
    return m_u0, u0

# ... [Splines] ...
def bspline(u, order = 6):
    #B-spline function
    if order == 6:
        u2 = u ** 2
        u3 = u ** 3
        u4 = u ** 4
        u5 = u ** 5
        u_less_1 = u - 1
        u_less_1_p5 = u_less_1 ** 5
        u_less_2 = u - 2
        u_less_2_p5 = u_less_2 ** 5
        u_less_3 = u - 3
        u_less_3_p5 = u_less_3 ** 5
        conditions = [
            torch.logical_and(u >= 0., u < 1.),
            torch.logical_and(u >= 1., u < 2.),
            torch.logical_and(u >= 2., u < 3.),
            torch.logical_and(u >= 3., u < 4.),
            torch.logical_and(u >= 4., u < 5.),
            torch.logical_and(u >= 5., u < 6.)
        ]
        outputs = [
            u5 / 120,
            u5 / 120 - u_less_1_p5 / 20,
            u5 / 120 + u_less_2_p5 / 8 - u_less_1_p5 / 20,
            u5 / 120 - u_less_3_p5 / 6 + u_less_2_p5 / 8 - u_less_1_p5 / 20,
            u5 / 24 - u4 + 19 * u3 / 2 - 89 * u2 / 2 + 409 * u / 4 - 1829 / 20,
            -u5 / 120 + u4 / 4 - 3 * u3 + 18 * u2 - 54 * u + 324 / 5
        ]
    return torch.sum(torch.stack([condition * output for condition, output in zip(conditions, outputs)]),axis=0)

def bspline_prime(u, order = 6):
    if order == 6:
        u2 = u ** 2
        u3 = u ** 3
        u4 = u ** 4
        u_less_1 = u - 1
        u_less_1_p4 = u_less_1 ** 4
        u_less_2 = u - 2
        u_less_2_p4 = u_less_2 ** 4
        conditions = [
            torch.logical_and(u >= 0., u < 1.),
            torch.logical_and(u >= 1., u < 2.),
            torch.logical_and(u >= 2., u < 3.),
            torch.logical_and(u >= 3., u < 4.),
            torch.logical_and(u >= 4., u < 5.),
            torch.logical_and(u >= 5., u < 6.)
        ]
        outputs = [
            u4 / 24,
            u4 / 24 - u_less_1_p4 / 4,
            u4 / 24 + 5 * u_less_2_p4 / 8 - u_less_1_p4 / 4,
            -5 * u4 / 12 + 6 * u3 - 63 * u2 / 2 + 71 * u - 231 / 4,
            5 * u4 / 24 - 4 * u3 + 57 * u2 / 2 - 89 * u + 409 / 4,
            -u4 / 24 + u3 - 9 * u2 + 36 * u - 54
        ]
    return torch.sum(torch.stack([condition * output for condition, output in zip(conditions, outputs)]),axis=0)

def bspline_prime2(u,order = 6):
    if order == 6:
        u2 = u ** 2
        u3 = u ** 3
        u_less_1 = u - 1
        conditions = [
            torch.logical_and(u >= 0., u < 1.),
            torch.logical_and(u >= 1., u < 2.),
            torch.logical_and(u >= 2., u < 3.),
            torch.logical_and(u >= 3., u < 4.),
            torch.logical_and(u >= 4., u < 5.),
            torch.logical_and(u >= 5., u < 6.)
        ]
        outputs = [
            u3 / 6,
            u3 / 6 - u_less_1 ** 3,
            5 * u3 / 3 - 12 * u2 + 27 * u - 19,
            -5 * u3 / 3 + 18 * u2 - 63 * u + 71,
            5 * u3 / 6 - 12 * u2 + 57 * u - 89,
            -u3 / 6 + 3 * u2 - 18 * u + 36
        ]
    return torch.sum(torch.stack([condition * output for condition, output in zip(conditions, outputs)]),axis=0)

def Q_mesh_on_m(Q_mesh_pera, m_u0, N, shifts):
    if isinstance(N, torch.Tensor):
        N = N.int()
    else:
        N = torch.tensor(N, dtype=torch.int64)

    indices_arr = (m_u0[:, None, :] + shifts) % N[None, None, :]
    indices_arr = indices_arr.to(dtype=torch.int64)

    Q_mesh = torch.zeros(N.tolist(), dtype=Q_mesh_pera.dtype)
    Q_mesh.index_put_(
        (
            indices_arr[:, :, 0].flatten(),
            indices_arr[:, :, 1].flatten(),
            indices_arr[:, :, 2].flatten()
        ),
        Q_mesh_pera.flatten(),
        accumulate=True
    )
    return Q_mesh

def setup_kpts_integer(N):
    N_half = N.reshape(3).tolist()
    kx, ky, kz = [torch.roll(torch.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2 ), - (N_half[i] - 1) // 2) for i in range(3)]
    kpts_int = torch.hstack([ki.flatten().unsqueeze(1) for ki in torch.meshgrid(kx, ky, kz, indexing='ij')])
    return kpts_int 

def setup_kpts(box, kpts_int):
    box_inv = torch.linalg.inv(box).T
    kpts_int = kpts_int.to(box_inv.dtype)
    kpts = 2 * torch.pi * torch.matmul(kpts_int, box_inv)
    ksq = torch.sum(kpts**2, axis=1)
    kpts = torch.hstack((kpts, ksq.unsqueeze(1))).T
    return kpts
def spread_Q(N, positions, box, Q_dict, n_mesh, rank):
    """
    Cartesian Spread Kernel (Updated for 6-component Quadrupoles)
    Uses Q_dict containing:
      - 'q': (N, 1)
      - 'p': (N, 3)
      - 'theta_vec': (N, 6) -> [xx, xy, xz, yy, yz, zz]
    """
    shifts = make_stencil(order=6)
    Nj_Aji_star = get_recip_vectors(N, box)
    m_u0, u0 = get_u_reference(positions, Nj_Aji_star)

    N_a = positions.shape[0]
    n_stencil = shifts.shape[1]

    # u shape: (N_a, n_stencil, 3)
    u = (u0[:, None, :] + shifts)
    M  = bspline(u)       # Value

    # --- Charge Term ---
    W_tot = Q_dict['q'] * torch.prod(M, dim=2)

    # --- Dipole Term ---
    if rank >= 1:
        dM = bspline_prime(u)

        # Rotate Dipole: p_grid = p_real * Recip_Vectors
        p_u = torch.matmul(Q_dict['p'], Nj_Aji_star.T)

        grad_W_u = torch.stack([
            dM[:,:,0] * M[:,:,1] * M[:,:,2], # du_x
            M[:,:,0] * dM[:,:,1] * M[:,:,2], # du_y
            M[:,:,0] * M[:,:,1] * dM[:,:,2]  # du_z
        ], dim=2)

        dip_term = torch.sum(p_u.unsqueeze(1) * grad_W_u, dim=2)
        W_tot -= dip_term

    # --- Quadrupole Term (6-Component Vector) ---
    if rank >= 2:
        d2M = bspline_prime2(u)

        # 1. Rotate the 6-component Quadrupole into Grid Basis
        # We need the full 3x3 rotation of the unique components.
        # It's actually cleaner to reconstruct the 3x3 locally for the rotation
        # OR rotate the tensor explicitly. Let's do the reconstruction method
        # as it is safer for generic triclinic boxes.

        # Unpack [xx, xy, xz, yy, yz, zz] -> 3x3
        t_vec = Q_dict['theta_vec'] # (N, 6)
        Theta_real = torch.zeros(N_a, 3, 3, device=u.device, dtype=u.dtype)

        # Fill diagonal
        Theta_real[:, 0, 0] = t_vec[:, 0] # xx
        Theta_real[:, 1, 1] = t_vec[:, 3] # yy
        Theta_real[:, 2, 2] = t_vec[:, 5] # zz
        trace_check = Theta_real[:,0,0] + Theta_real[:,1,1] + Theta_real[:,2,2]
        print(f"Mean Trace: {trace_check.mean().item()}")
        print(f"Max Trace:  {trace_check.abs().max().item()}")

        # Fill off-diagonal (Symmetric)
        Theta_real[:, 0, 1] = t_vec[:, 1]; Theta_real[:, 1, 0] = t_vec[:, 1] # xy
        Theta_real[:, 0, 2] = t_vec[:, 2]; Theta_real[:, 2, 0] = t_vec[:, 2] # xz
        Theta_real[:, 1, 2] = t_vec[:, 4]; Theta_real[:, 2, 1] = t_vec[:, 4] # yz

        # Rotate: T_u = R * T_real * R.T
        Theta_u = torch.einsum('im,jn,kmn->kij', Nj_Aji_star, Nj_Aji_star, Theta_real)
        #2. Spline Hessians
        H_xx = d2M[:,:,0] * M[:,:,1] * M[:,:,2]
        H_yy = M[:,:,0] * d2M[:,:,1] * M[:,:,2]
        H_zz = M[:,:,0] * M[:,:,1] * d2M[:,:,2]
        H_xy = dM[:,:,0] * dM[:,:,1] * M[:,:,2]
        H_xz = dM[:,:,0] * M[:,:,1] * dM[:,:,2]
        H_yz = M[:,:,0] * dM[:,:,1] * dM[:,:,2]
        # 3. Contraction (Explicit Summation)
        # Sum = T_xx*H_xx + T_yy*H_yy + T_zz*H_zz + 2*(T_xy*H_xy + ...)

        term_diag = (Theta_u[:, None, 0, 0] * H_xx +
                     Theta_u[:, None, 1, 1] * H_yy +
                     Theta_u[:, None, 2, 2] * H_zz)

        term_off  = 2.0 * (Theta_u[:, None, 0, 1] * H_xy +
                           Theta_u[:, None, 0, 2] * H_xz +
                           Theta_u[:, None, 1, 2] * H_yz)

        quad_term = term_diag + term_off

        scale = 0.5
        W_tot += scale * quad_term

    # Scatter to grid
    Q_mesh = Q_mesh_on_m(W_tot, m_u0, N, shifts)
#    print("CARTESIAN SPREAD")
#    print(f"Total Sum of Grid: {torch.sum(Q_mesh).item()}") # Should be close to sum(q)
#    print(f"Max Value in Grid: {torch.max(torch.abs(Q_mesh)).item()}")
#    print(f"Q_mesh shape: {Q_mesh.shape}")
#    print("CARTESIAN Q_MESH[0]")
#    print(Q_mesh[0])
#    print("CARTESIAN Q_MESH[1]")
#    print(Q_mesh[1])
#    print("CARTESIAN Q_MESH[2]")
#    print(Q_mesh[2])
    return Q_mesh
def Ck_1(ksq, kappa, V):
    return 4*torch.pi/V/ksq * torch.exp(-ksq/4/kappa**2)

def get_pme_recip(Ck_fn, kappa,positions,box,Q_dict,K1,K2,K3,rank, bspline_order = 6):
    bspline_range = torch.arange(-bspline_order//2, bspline_order//2)
    shifts  = make_stencil(order=6)
    
    # spread Q
    N = torch.tensor([K1, K2, K3])
    n_mesh = shifts.shape[1]
    
    # CALL CARTESIAN SPREAD
    Q_mesh = spread_Q(N,positions, box, Q_dict, n_mesh, rank)
    
    N = N.reshape((1, 1, 3))
    kpts_int = setup_kpts_integer(N)
    kpts = setup_kpts(box, kpts_int)
    half   = bspline_order // 2                    
    m = torch.arange(-half, half).reshape(-1, 1, 1)              
    theta_k = torch.prod(
            torch.sum(
                bspline(m + bspline_order/2) * torch.cos(2*torch.pi*m*kpts_int.unsqueeze(0) / N),
                axis = 0
                ),
            axis = 1
            )
    V = torch.linalg.det(box)
    S_k = torch.fft.fftn(Q_mesh).flatten()
    C_k = Ck_fn(kpts[3,1:], kappa, V)
    Phi_k = torch.zeros_like(S_k)
    Phi_k[1:] = C_k*S_k[1:]/torch.abs(theta_k[1:])**2
    Phi_k_3d = Phi_k.reshape(K1,K2,K3)
    Phi_real_space = torch.fft.ifftn(Phi_k_3d, norm='forward').real
    #Energy straight from grid. Useful for debugging
    #E_k = 0.5 * torch.sum(C_k * torch.abs(S_k[1:] / theta_k[1:])**2) 
    #print(f"CARTESIAN PME ENERGY FROM STRUCTURE FACTOR: {E_k}")

    # Interpolate Back
    long_range_potential, long_range_field, long_range_field_gradient = interpolate_to_atoms(
        Phi_real_space, positions, box, N, rank
    )
    
    return long_range_potential, long_range_field, long_range_field_gradient 
def construct_Q(q, p, t, rank):
    """
    Args:
        q: (N, 1)
        p: (N, 3)
        t: (N, 6) Quadrupoles [xx, xy, xz, yy, yz, zz]
    """
    N_a = q.shape[0]
    Q_dict = {'q': q.reshape(N_a, 1)}

    if rank >= 1:
        Q_dict['p'] = p.reshape(N_a, 3)

    if rank >= 2:
        # Enforce Tracelessness just in case
        # Trace = Qxx + Qyy + Qzz
        trace = t[:, 0] + t[:, 3] + t[:, 5]

        # Check if trace is effectively zero
        if torch.any(torch.abs(trace) > 1e-5):
            print("WARNING: Input Quadrupoles are not traceless. Removing trace...")
            t_clean = t.clone()
            correction = trace / 3.0
            t_clean[:, 0] -= correction
            t_clean[:, 3] -= correction
            t_clean[:, 5] -= correction
            Q_dict['theta_vec'] = t_clean
        else:
            Q_dict['theta_vec'] = t

    return Q_dict


def make_stencil(order: int):
    half = order // 2
    r     = torch.arange(-half, half)          
    shifts = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1)
    shifts = shifts.reshape(1, order**3, 3)                   
    return shifts                                     

def interpolate_to_atoms(phi_grid: torch.Tensor,
                          positions: torch.Tensor,
                          box: torch.Tensor,
                          N: torch.Tensor,
                          rank) -> torch.Tensor:

    """
    Cartesian Interpolation.
    Calculates Potential, Field, and Gradient by taking derivatives of B-splines.
    """
    order = 6
    Nj_Aji_star = get_recip_vectors(N, box)                 
    m_u0, u0    = get_u_reference(positions, Nj_Aji_star)   
    shifts      = make_stencil(order)
    n_mesh      = order**3
    Na          = positions.shape[0]

    # Reciprocal Space "Metric" for chain rule
    # grad_r = M * grad_u  => M = Nj_Aji_star
    M_mat = Nj_Aji_star # (3, 3)

    # 1. Gather Grid Values
    if N.dim() > 1: N = N.flatten()
    Nx, Ny, Nz = N.tolist() 
    m_idx = (m_u0[:, None, :] + shifts[0]) % N[None, None, :]
    flat   = phi_grid.reshape(-1)    
    grid_i = (m_idx[..., 0] * Ny * Nz  +
              m_idx[..., 1] * Nz +
              m_idx[..., 2])                                
    phi_loc = flat[grid_i] # Shape (Na, n_stencil)

    # 2. Evaluate Splines
    u = (u0[:, None, :] + shifts) 
    B  = bspline(u)       # (Na, S, 3)
    
    # --- Potential (Phi) ---
    # Sum( GridVal * Bx * By * Bz )
    W_base = torch.prod(B, dim=2)
    phi_atoms = torch.sum(phi_loc * W_base, dim=1)

    E_atoms, EG_atoms = None, None

    # --- Field (E = - grad Phi) ---
    if rank >= 1:
        dB = bspline_prime(u)
        
        # Gradient in u-space
        grad_W_u = torch.stack([
            dB[:,:,0] * B[:,:,1] * B[:,:,2], 
            B[:,:,0] * dB[:,:,1] * B[:,:,2], 
            B[:,:,0] * B[:,:,1] * dB[:,:,2]
        ], dim=2) 
        
        # Interpolate Gradient in u-space
        # grad_phi_u = Sum( phi_loc * grad_W_u )
        grad_phi_u = torch.sum(phi_loc.unsqueeze(-1) * grad_W_u, dim=1) # (Na, 3)
        
        # Transform to Real Space: grad_r = M * grad_u
        # E = - grad_r
        E_atoms = torch.matmul(grad_phi_u, M_mat.T) # Note: M_mat is (3,3) Recip

    # --- Field Gradient (grad E = - hess Phi) ---
    if rank >= 2:
        d2B = bspline_prime2(u)
        
        # Hessian in u-space
        H_xx = d2B[:,:,0] * B[:,:,1] * B[:,:,2]
        H_yy = B[:,:,0] * d2B[:,:,1] * B[:,:,2]
        H_zz = B[:,:,0] * B[:,:,1] * d2B[:,:,2]
        H_xy = dB[:,:,0] * dB[:,:,1] * B[:,:,2]
        H_xz = dB[:,:,0] * B[:,:,1] * dB[:,:,2]
        H_yz = B[:,:,0] * dB[:,:,1] * dB[:,:,2]
        
        hess_W_u = torch.zeros(Na, n_mesh, 3, 3, device=u.device, dtype=u.dtype)
        hess_W_u[:,:,0,0] = H_xx
        hess_W_u[:,:,1,1] = H_yy
        hess_W_u[:,:,2,2] = H_zz
        hess_W_u[:,:,0,1] = H_xy; hess_W_u[:,:,1,0] = H_xy
        hess_W_u[:,:,0,2] = H_xz; hess_W_u[:,:,2,0] = H_xz
        hess_W_u[:,:,1,2] = H_yz; hess_W_u[:,:,2,1] = H_yz
        # Interpolate Hessian in u-space
        hess_phi_u = torch.sum(phi_loc.reshape(Na, n_mesh, 1, 1) * hess_W_u, dim=1) # (Na, 3, 3)
        # Transform to Real Space: Hess_r = M * Hess_u * M.T
        # Einstein: H_r_ij = M_im * H_u_mn * M_jn
        grad_E = torch.einsum('mi,knm,nj->kij', M_mat, hess_phi_u, M_mat) / 2.0
        EG_atoms = -grad_E

    return phi_atoms, E_atoms, EG_atoms
def compute_pme(coords, box, q, p, t, kappa, k_max, rank):
    K1, K2, K3 = k_max, k_max, k_max

    # 1. Setup Data
    Q_dict = construct_Q(q, p, t, rank)

    # 2. Reciprocal Space Calculation
    # Note: get_pme_recip internally calls the NEW spread_Q
    long_range_potential, long_range_field, long_range_field_gradient= get_pme_recip(
        Ck_1, kappa, coords, box, Q_dict, K1, K2, K3, rank, bspline_order=6
    )

    # 3. Self Corrections
    kappa_val = kappa
    kappa_over_root_pi = kappa_val / math.sqrt(math.pi)

    # Potential Correction
    # E_self = E_recip - E_real. We subtract self energy.
    # Monopole Self
    self_potential = - 2 * kappa_over_root_pi * q
    print(f"CARTESIAN PME  Atom 0 Potential Self Correction: {self_potential[0].tolist()}") 
    long_range_potential += self_potential

    if rank == 0: return long_range_potential

    # Field Correction (Dipole)
    # 4*k^3 / 3*sqrt(pi)
    field_correction = kappa_over_root_pi * (4 * kappa_val**2 / 3.0) * p
    print(f"CARTESIAN PME  Atom 0 Field Self Correction: {field_correction[0].tolist()}") 
    long_range_field += field_correction

    if rank == 1: return long_range_potential, long_range_field

    # Gradient Correction (Quadrupole)
    # For Quadrupoles, we need to be careful with the self-energy definition.
    # If t is [xx, xy, xz, yy, yz, zz], the self energy usually involves
    # a contraction Q:Q.

    # Factor: 16 * k^5 / 15 * sqrt(pi) * Q

    t_vec = Q_dict['theta_vec']
    t_full = torch.zeros(t_vec.shape[0], 3, 3, device=t_vec.device)
    t_full[:,0,0] = t_vec[:,0]; t_full[:,1,1] = t_vec[:,3]; t_full[:,2,2] = t_vec[:,5]
    t_full[:,0,1] = t_vec[:,1]; t_full[:,1,0] = t_vec[:,1]
    t_full[:,0,2] = t_vec[:,2]; t_full[:,2,0] = t_vec[:,2]
    t_full[:,1,2] = t_vec[:,4]; t_full[:,2,1] = t_vec[:,4]

    grad_correction = kappa_over_root_pi * (16 * kappa_val**4 / 15.0) * t_full
    print(f"CARTESIAN PME  Atom 0 Field Grad Self Correction: {grad_correction[0].tolist()}")
    long_range_field_gradient += grad_correction

    return long_range_potential, long_range_field, long_range_field_gradient

