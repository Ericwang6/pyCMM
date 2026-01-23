import torch
from scipy import special
from torch.special import erfc
import math
from . import units, pbc
from .units import HARTREE2KCAL, BOHR2ANG 
from .pbc import applyPBC
import numpy as np
from itertools import combinations

torch.set_printoptions(profile="full")
diff_flag = 1
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
    
#    # --- DEBUG: Match CUDA [GPU] Coords and Mapping ---
#    print(f"\n[PY] Mapping Coords[0]: {coords[0].tolist()}")
#    print(f"[PY] Atom 0 Grid Mapping: m_u0={m_u0[0].tolist()}, u_frac={u0[0].tolist()}")
#    # --------------------------------------------------
    
    return m_u0, u0

# ... [Splines] ...
def b5spline(u, order=5):
    #B-spline order 5
    if order == 5:
        u2 = u ** 2
        u3 = u ** 3
        u4 = u ** 4
        conditions = [
            (u >= 0.) & (u < 1.),
            (u >= 1.) & (u < 2.),
            (u >= 2.) & (u < 3.),
            (u >= 3.) & (u < 4.),
            (u >= 4.) & (u < 5.)
        ]
        outputs = [
            u4 / 24,
            -u4/6 + 5*u3/6 -5*u2/4 + 5*u/6 - 5/24,
            u4/4 - 5*u3/2 + 35*u2/4 - 25*u/2 + 155/24,
            -u4/6 + 5*u3/2 - 55*u2/4 + 65*u/2 - 655/24,
            ((5 - u) ** 4) / 24
        ]
    return torch.sum(torch.stack([cond * out for cond, out in zip(conditions, outputs)]), axis=0)

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

def get_theta(u,M_u):
    theta = torch.prod(M_u,axis=-1)
    return theta

def get_thetaprime(u,Nj_Aji_star, M_u,Mprime_u):
    div = torch.stack([
                Mprime_u[:, 0] * M_u[:, 1] * M_u[:, 2],
                Mprime_u[:, 1] * M_u[:, 2] * M_u[:, 0],
                Mprime_u[:, 2] * M_u[:, 0] * M_u[:, 1],
            ]).T
    return torch.einsum("ij,kj->ki",-Nj_Aji_star,div)

def get_theta2prime(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u):
    div_00 = M2prime_u[:, 0] * M_u[:, 1] * M_u[:, 2]
    div_11 = M2prime_u[:, 1] * M_u[:, 0] * M_u[:, 2]
    div_22 = M2prime_u[:, 2] * M_u[:, 0] * M_u[:, 1]
    div_01 = Mprime_u[:, 0] * Mprime_u[:, 1] * M_u[:, 2]
    div_02 = Mprime_u[:, 0] * Mprime_u[:, 2] * M_u[:, 1]
    div_12 = Mprime_u[:, 1] * Mprime_u[:, 2] * M_u[:, 0]
    div_10 = div_01
    div_20 = div_02
    div_21 = div_12
    div = torch.stack([
        torch.stack([div_00, div_01, div_02]),
        torch.stack([div_10, div_11, div_12]),
        torch.stack([div_20, div_21, div_22]),
    ]).permute(2,0,1)
    return torch.einsum("im,jn,kmn->kij", -Nj_Aji_star, -Nj_Aji_star, div)

def sph_harmonics_GO(u0, Nj_Aji_star,shifts,n_mesh, rank):
    n_harm = int((rank + 1)**2)
    N_a = u0.shape[0]
    u = (u0[:, None, :] + shifts).reshape((N_a * n_mesh, 3))
    M_u = bspline(u)
    theta = get_theta(u, M_u)
    if rank == 0:
        return theta.reshape(N_a, n_mesh, n_harm)
    Mprime_u = bspline_prime(u)
    thetaprime = get_thetaprime(u, Nj_Aji_star, M_u, Mprime_u)
    harmonics_1 = torch.stack(
        [theta,
        thetaprime[:, 2],
        thetaprime[:, 0],
        thetaprime[:, 1]],
        axis = -1
    )
    if rank == 1:
        return harmonics_1.reshape(N_a, n_mesh, n_harm)
    M2prime_u = bspline_prime2(u)
    theta2prime = get_theta2prime(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u)
    rt3 = np.sqrt(3)
    theta2prime_trace = torch.einsum("ijj->i", theta2prime)
    harmonics_2 = torch.hstack(
        [harmonics_1,
        torch.stack([(3*theta2prime[:,2,2] - theta2prime_trace)/2,
        rt3 * theta2prime[:, 0, 2],
        rt3 * theta2prime[:, 1, 2],
        rt3/2 * (theta2prime[:, 0, 0] - theta2prime[:, 1, 1]),
        rt3 * theta2prime[:, 0, 1]], axis = 1)]
    )
    if rank == 2:
        return harmonics_2.reshape(N_a, n_mesh, n_harm)
    else:
        raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

def Q_m_peratom(Q, sph_harms,n_mesh, rank):
    N_a = sph_harms.shape[0] 
    Q_dbf = Q[:, 0:1]
    if rank >= 1:
        Q_dbf = torch.hstack([Q_dbf, Q[:,1:4]])
    if rank >= 2:
        Q_dbf = torch.hstack([Q_dbf, Q[:,4:9]/3])
    Q_m_pera = torch.sum(Q_dbf.unsqueeze(1) * sph_harms, dim=2)
    assert Q_m_pera.shape == (N_a, n_mesh)
    return Q_m_pera

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

def spread_Q(N, positions, box, Q, n_mesh, rank):
    shifts = make_stencil(order=6)
    Nj_Aji_star = get_recip_vectors(N, box)

    # 1. Map to Grid
    m_u0, u0 = get_u_reference(positions, Nj_Aji_star)
    sph_harms = sph_harmonics_GO(u0, Nj_Aji_star, shifts, n_mesh, rank)
    Q_mesh_pera = Q_m_peratom(Q, sph_harms, n_mesh, rank)

    Q_mesh = Q_mesh_on_m(Q_mesh_pera, m_u0, N, shifts)

#    print("SPHERICAL SPREAD")
#    print(f"Q_mesh shape: {Q_mesh.shape}")
#    print(f"Total Sum of Grid: {torch.sum(Q_mesh).item()}") # Should be close to sum(q)
#    print(f"Max Value in Grid: {torch.max(torch.abs(Q_mesh)).item()}")
#    print("SPHERICAL SPREAD Q_MESH[0]")
#    print(Q_mesh[0])
#    print("SPHERICAL SPREAD Q_MESH[1]")
#    print(Q_mesh[1])
#    print("SPHERICAL SPREAD Q_MESH[2]")
#    print(Q_mesh[2])


    return Q_mesh

def Ck_1(ksq, kappa, V):
    return 4*torch.pi/V/ksq * torch.exp(-ksq/4/kappa**2)

def get_pme_recip(Ck_fn, kappa,positions,box,Q,K1,K2,K3,rank, bspline_order = 6):
    bspline_range = torch.arange(-bspline_order//2, bspline_order//2)
    n_mesh = (bspline_order)**3
    shifts  = make_stencil(order=6)
    # spread Q
    N = torch.tensor([K1, K2, K3])
    N_ = torch.tensor([K1, K2, K3])
    n_mesh = shifts.shape[1]
    Q_mesh = spread_Q(N,positions, box, Q,n_mesh,rank)
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
    E_k = 0.5 * torch.sum(C_k * torch.abs(S_k[1:] / theta_k[1:])**2) 
    print(f"SPHERICAL PME ENERGY FROM STRUCTURE FACTOR: {E_k}")
    
    E_grid,EG_grid = 0,0 
    if rank == 0:
        long_range_potential, long_range_field, long_range_field_gradient= interpolate_to_atoms(Phi_real_space,E_grid,EG_grid,positions,box,N_,rank)
    if diff_flag == 2:
        if rank >= 1:
            kx = kpts[0].reshape(K1, K2, K3)
            ky = kpts[1].reshape(K1, K2, K3)
            kz = kpts[2].reshape(K1, K2, K3)
            E_kx = 1j * kx * Phi_k_3d
            E_ky = 1j * ky * Phi_k_3d
            E_kz = 1j * kz * Phi_k_3d
            E_grid_x = torch.fft.ifftn(E_kx, norm='forward').real
            E_grid_y = torch.fft.ifftn(E_ky, norm='forward').real
            E_grid_z = torch.fft.ifftn(E_kz, norm='forward').real
            E_grid = torch.stack([E_grid_z, E_grid_x, E_grid_y], dim=-1)
        if rank>=2:
            kx2 = kx**2
            ky2 = ky**2
            kz2 = kz**2
            kxy = kx * ky
            kxz = kx * kz
            kyz = ky * kz
            sqrt3 = torch.sqrt(torch.tensor(3.0))
            Phi_k_20  = 0.5 * (2 * kz2 - kx2 - ky2) * Phi_k_3d
            Phi_k_21c = sqrt3 * kxz * Phi_k_3d
            Phi_k_21s = sqrt3 * kyz * Phi_k_3d
            Phi_k_22c = 0.5 * sqrt3* (kx2 - ky2) * Phi_k_3d
            Phi_k_22s =    sqrt3 * kxy * Phi_k_3d
            EG_20  = -torch.fft.ifftn(Phi_k_20,  norm='forward').real
            EG_21c = -torch.fft.ifftn(Phi_k_21c, norm='forward').real
            EG_21s = -torch.fft.ifftn(Phi_k_21s, norm='forward').real
            EG_22c = -torch.fft.ifftn(Phi_k_22c, norm='forward').real
            EG_22s = -torch.fft.ifftn(Phi_k_22s, norm='forward').real
            EG_grid = torch.stack([EG_20, EG_21c, EG_21s, EG_22c, EG_22s], dim=-1)
        long_range_potential, long_range_field, long_range_field_gradient= interpolate_to_atoms(Phi_real_space,E_grid,EG_grid,positions,box,N_, rank)
    else:
        E_grid, EG_grid = 0,0
        long_range_potential, long_range_field, long_range_field_gradient= interpolate_to_atoms(Phi_real_space, E_grid, EG_grid, positions, box, N_, rank)
    return long_range_potential, long_range_field, long_range_field_gradient,E_k

def construct_Q(q, p, t, rank):
    #print(f"\n[DEBUG construct_Q] Rank={rank}")
    #print(f"  Input shapes - q: {q.shape}, p: {p.shape}, t: {t.shape}")
    N_a = q.shape[0]
    q = q.reshape(N_a, 1)
    if rank == 0:
        return q
    p = p.reshape(N_a, 3)
    if rank == 1:
        return torch.hstack([q, p])
    if rank == 2:
        if t.abs().sum() == 0:
             print("  [CRITICAL WARNING] 't' tensor passed to construct_Q is ALL ZEROS!")
        try:
            t_vec = t.reshape(N_a, 5)
        except RuntimeError as e:
            print(f"  [ERROR] Reshape failed. 't' shape is {t.shape}, cannot reshape to ({N_a}, 5)")
            raise e
        Q = torch.hstack([q, p, t_vec])
        Q = Q.contiguous()
        return Q

def make_stencil(order: int):
    half = order // 2
    r     = torch.arange(-half, half)          
    shifts = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1)
    shifts = shifts.reshape(1, order**3, 3)                   
    return shifts                                             

def interpolate_to_atoms(phi_grid: torch.Tensor,
                         E_grid: torch.Tensor,
                         EG_grid: torch.Tensor,
                         positions: torch.Tensor,
                         box: torch.Tensor,
                         N: torch.Tensor,
                         rank) -> torch.Tensor:

    """
    B‑spline interpolation consistent with spread_Q().
    """
    #print("\n[PY] Interpolate Kernel Started for Atom 0")

    order = 6
    E_atoms,EG_atoms = 0,0
    Nj_Aji_star = get_recip_vectors(N, box)                 
    m_u0, u0    = get_u_reference(positions, Nj_Aji_star)   
    shifts      = make_stencil(order)
    n_mesh      = order**3
    Na          = positions.shape[0]

    stgo = sph_harmonics_GO(u0, Nj_Aji_star, shifts, n_mesh,rank)                           
    theta = stgo[...,0]

    Nx, Ny, Nz = N.tolist() 
    m_idx = (m_u0[:, None, :] + shifts[0]) % N[None, None, :]
    flat   = phi_grid.reshape(-1)    
    grid_i = (m_idx[..., 0] * Ny * Nz  +
              m_idx[..., 1] * Nz +
              m_idx[..., 2])                                
    phi_loc = flat[grid_i]  
    phi_atoms = (theta * phi_loc).sum(dim=1)                
    E_atoms, EG_atoms = None, None
    if diff_flag == 2:
        if rank >=1:
            E_grid_flat = E_grid.reshape(-1, 3)
            E_stencil = E_grid_flat[grid_i]
            E_atoms = torch.sum(theta.unsqueeze(-1) * E_stencil, dim=1)
        if rank>=2:
            EG_flat = EG_grid.reshape(-1, 5)
            EG_stencil = EG_flat[grid_i]
            theta /= 3.0
            EG_atoms = torch.sum(theta.unsqueeze(-1) * EG_stencil, dim=1)
    else:  
        if rank>=1:
            # sph_harmonics_GO rank 1 returns [theta, dTheta_z, dTheta_x, dTheta_y]
            thetaprime_z = stgo[...,1]
            thetaprime_x = stgo[...,2]
            thetaprime_y = stgo[...,3]
            E_x = (thetaprime_x * phi_loc).sum(dim=1)  
            E_y = (thetaprime_y * phi_loc).sum(dim=1) 
            E_z = (thetaprime_z * phi_loc).sum(dim=1)
            E_atoms = -torch.stack([E_z,E_x,E_y],dim=1)

        if rank>=2:
            thetaprime2 = stgo[...,4:9] /3.0 
            thetaprime2_1 = thetaprime2[...,0]   
            thetaprime2_2 = thetaprime2[...,1]   
            thetaprime2_3 = thetaprime2[...,2] 
            thetaprime2_4 = thetaprime2[...,3] 
            thetaprime2_5 = thetaprime2[...,4] 
            EG_1 = -(thetaprime2_1 * phi_loc).sum(dim=1)
            EG_2 = -(thetaprime2_2 * phi_loc).sum(dim=1)
            EG_3 = -(thetaprime2_3 * phi_loc).sum(dim=1)
            EG_4 = -(thetaprime2_4 * phi_loc).sum(dim=1)
            EG_5 = -(thetaprime2_5 * phi_loc).sum(dim=1)
            EG_atoms = torch.stack([EG_1,EG_2,EG_3,EG_4,EG_5], dim=1)
    return phi_atoms, E_atoms, EG_atoms


def compute_pme(coords, box, q ,p, t, kappa, k_max ,rank):
  K1,K2,K3 = k_max, k_max, k_max
  Q = construct_Q(q,p,t, rank)
  long_range_potential, long_range_field, long_range_field_gradient, E_k = get_pme_recip(Ck_1, kappa,coords,box,Q,K1,K2,K3, rank, bspline_order=6)
  
  # Self Corrections
  kappa_over_root_pi = kappa / torch.sqrt(torch.tensor(torch.pi))
  
  # 1. Potential Self Correction
  potential_correction = -2* kappa_over_root_pi * q
  print(f"SPHERICAL PME  Atom 0 Potential Self Correction: {potential_correction[0].tolist()}") 
  long_range_potential += potential_correction
  
  if rank == 0:
      return long_range_potential
      
  # 2. Field Self )
  field_correction = kappa_over_root_pi * (4 * kappa * kappa / 3) * p
  print(f"SPHERICAL PME  Atom 0 Field Self Correction: {field_correction[0].tolist()}") 
  long_range_field += field_correction

  if rank == 1:
      return long_range_potential, long_range_field
      
  # 3. Gradient Self Correction
  grad_correction = (2/3)* kappa_over_root_pi * (16 * kappa * kappa * kappa * kappa / 5) * t / 3
  print(f"SPHERICAL PME Atom 0 Field Grad Self Correction: {grad_correction[0].tolist()}")
  long_range_field_gradient += grad_correction
  
  if rank == 2:
      return long_range_potential, long_range_field, long_range_field_gradient, E_k
