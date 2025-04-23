import torch
from scipy import special
from torch.special import erfc
import math
from . import units, pbc
from .units import EPSILON0, HARTREE2KCAL, BOHR2ANG 
from .pbc import applyPBC
import numpy as np
from itertools import combinations
torch.set_printoptions(profile="full")
lmax = 2
diff_flag = 1
#This file contains the Ewald Summation for computing Long range interactions
#Multipolar Ewald Methods, 1: Theory, Accuracy, and Performance
#Timothy J. Giese, Maria T. Panteva, Haoyuan Chen, and Darrin M. York Journal of Chemical Theory and Computation 2015 11 (2), 436-450 DOI: 10.1021/ct5007983
def self_interaction_vectorized(coords,q,p,t,kappa):
    U_q = torch.dot(q,q) * kappa/math.sqrt(torch.pi)
    U_d = torch.sum(p*p) * 2 * kappa**3 /(3 * math.sqrt(torch.pi))
    U_cq = torch.dot(q,torch.einsum("bii->b",t)) * 2*kappa**3/(3 * torch.pi)
    U_t = torch.sum(t*t) * 8*kappa**5/(45*torch.pi)
    #total self-interaction energy
    U_self = U_q + U_d + U_cq + U_t
    return U_self
########################################################################################################################
def get_recip_vectors(N,box):
    """
    Get recip lattice vectors of grid
    Input:
        N: (3,) array 
        box: 3x3 matrix of real space box
    Output:
        Nh_Aji_star : 3x3 matrix of recip lattice vectors
    """
    Nj_Aji_star = (N.reshape((1,3)) * torch.linalg.inv(box)).T

    return Nj_Aji_star
def get_u_reference(coords, Nj_Aji_star):
    """
    Maps particle positions to grid
    Output:
        m_u0: nearest grid point for each particle
        u0: fractional displacement within the grid
    """
    bspline_order = 6
    R_in_m_basis = torch.einsum("ij,kj->ki",Nj_Aji_star,coords)
    m_u0 = torch.ceil(R_in_m_basis).to(torch.int64)
    u0 = (m_u0 - R_in_m_basis) + bspline_order/2
    return m_u0, u0

def b5spline(u, order=5):
    #B-spline order 5. TO check with Q-Chem implementation!
    #DERIVATIVES OF THIS HAVE NOT BEEN IMPLEMENTED YET
    if order == 5:
        u2 = u ** 2
        u3 = u ** 3
        u4 = u ** 4
        u_less_1 = u - 1
        u_less_1_p4 = u_less_1 ** 4
        u_less_2 = u - 2
        u_less_2_p4 = u_less_2 ** 4
        u_less_3 = u - 3
        u_less_3_p4 = u_less_3 ** 4

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
    #First order derivative of B-spline function
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
    #Second order derivative of B-spline function
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
    #evaluate B-spline weight theta
    theta = torch.prod(M_u,axis=-1)
    return theta

def get_thetaprime(u,Nj_Aji_star, M_u,Mprime_u):
    #Evaluated first order derivative of theta
    div = torch.stack([
                Mprime_u[:, 0] * M_u[:, 1] * M_u[:, 2],
                Mprime_u[:, 1] * M_u[:, 2] * M_u[:, 0],
                Mprime_u[:, 2] * M_u[:, 0] * M_u[:, 1],
            ]).T
    return torch.einsum("ij,kj->ki",-Nj_Aji_star,div)

def get_theta2prime(u, Nj_Aji_star, M_u, Mprime_u, M2prime_u):
    """
    compute the 3 x 3 second derivatives of theta with respect to xyz
    
    Input:
        u
        Nj_Aji_star
    
    Output:
        N_A * 3 * 3
    """
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
    # Notice that u = m_u0 - R_in_m_basis + 6/2
    # therefore the Jacobian du_j/dx_i = - Nj_Aji_star
    return torch.einsum("im,jn,kmn->kij", -Nj_Aji_star, -Nj_Aji_star, div)
def sph_harmonics_GO(u0, Nj_Aji_star,shifts,n_mesh):
    '''
    Find out the value of spherical harmonics GRADIENT OPERATORS, assume the order is:
    00, 10, 11c, 11s, 20, 21c, 21s, 22c, 22s, ...
    Currently supports lmax <= 2    
    In other words: this returns the b-spline weights of a meshed particle point around its
    stencil(subgrid) in the main grid.
    Inputs:
        u0: 

            reciprocal lattice vectors in the m-grid
        lmax:
            int: max L

    Output: 
        harmonics: 
            a Na * (6**3) * (l+1)^2 matrix, STGO operated on theta,
            evaluated at 6*6*6 integer points about reference points m_u0 
    '''
    n_harm = int((lmax + 1)**2)
    N_a = u0.shape[0]
    ## mesh points around each site
    u = (u0[:, None, :] + shifts).reshape((N_a * n_mesh, 3))
    M_u = bspline(u)
    theta = get_theta(u, M_u)
    if lmax == 0:
        return theta.reshape(N_a, n_mesh, n_harm)
    ## dipole
    Mprime_u = bspline_prime(u)
    thetaprime = get_thetaprime(u, Nj_Aji_star, M_u, Mprime_u)
    harmonics_1 = torch.stack(
        [theta,
        thetaprime[:, 2],
        thetaprime[:, 0],
        thetaprime[:, 1]],
        axis = -1
    )
    if lmax == 1:
        return harmonics_1.reshape(N_a, n_mesh, n_harm)
    ## quadrapole
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
    if lmax == 2:
        return harmonics_2.reshape(N_a, n_mesh, n_harm)
    else:
        raise NotImplementedError('l > 2 (beyond quadrupole) not supported')
def Q_m_peratom(Q, sph_harms,n_mesh):
    """
    Computes <R_t|Q>.Essential for calculation structure factors. See eq. (49) of https://doi.org/10.1021/ct5007983
    
    Inputs:
        Q: 
            N_a * (l+1)**2 matrix containing global frame multipole moments up to lmax,
        sph_harms:
            N_a, 216, (l+1)**2
        lmax:
            int: maximal L
    
    Output:
        Q_m_pera:
            N_a * 216 matrix, values of theta evaluated on a 6 * 6 block about the atoms
    """
    N_a = sph_harms.shape[0] 
    Q_dbf = Q[:, 0:1]
    if lmax >= 1:
        Q_dbf = torch.hstack([Q_dbf, Q[:,1:4]])
    if lmax >= 2:
        Q_dbf = torch.hstack([Q_dbf, Q[:,4:9]/3])#Q[:9] in DMFF as their quadrupole moment is defined using only 5 values
    Q_m_pera = torch.sum(Q_dbf.unsqueeze(1) * sph_harms, dim=2)
    assert Q_m_pera.shape == (N_a, n_mesh)
    return Q_m_pera
def Q_mesh_on_m(Q_mesh_pera, m_u0, N, shifts):
    """
    Reduce the local Q_m_peratom into the global mesh
    """
    # Ensure torch types
    if isinstance(N, torch.Tensor):
        N = N.int()
    else:
        N = torch.tensor(N, dtype=torch.int64)

    indices_arr = (m_u0[:, None, :] + shifts) % N[None, None, :]
    indices_arr = indices_arr.to(dtype=torch.int64)

    Q_mesh = torch.zeros(N.tolist(), dtype=Q_mesh_pera.dtype, device=Q_mesh_pera.device)
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
    """
    Outputs:
        kpts_int:
            n_k * 3 matrix, n_k = N[0] * N[1] * N[2]
    """
    N_half = N.reshape(3).tolist()
    kx, ky, kz = [torch.roll(torch.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2 ), - (N_half[i] - 1) // 2) for i in range(3)]
    kpts_int = torch.hstack([ki.flatten().unsqueeze(1) for ki in torch.meshgrid(kx, ky, kz, indexing='ij')])
    return kpts_int 
def setup_kpts(box, kpts_int):
    '''
    This function sets up the k-points used for reciprocal space calculations
    
    Input:
        box:
            3 * 3, three axis arranged in rows
        kpts_int:
            n_k * 3 matrix

    Output:
        kpts:
            4 * K, K=K1*K2*K3, contains kx, ky, kz, k^2 for each kpoint
    '''
    # in this array, a*, b*, c* (without 2*pi) are arranged in column
    box_inv = torch.linalg.inv(box).T
    kpts_int = kpts_int.to(box_inv.dtype)
    # K * 3, coordinate in reciprocal space
    kpts = 2 * torch.pi * torch.matmul(kpts_int, box_inv)
    ksq = torch.sum(kpts**2, axis=1)
    kpts = torch.hstack((kpts, ksq.unsqueeze(1))).T
    return kpts
def spread_Q(N,positions, box, Q,n_mesh):
    '''
    This is the high level wrapper function, in charge of spreading the charges/multipoles on grid

    Input:
        positions:
            Na * 3: positions of each site
        box: 
            3 * 3: box
        Q:
            Na * (lmax+1)**2: the multipole of each site in global frame

    Output:
        Q_mesh:
            K1 * K2 * K3: the meshed multipoles
        
    '''
    shifts  = make_stencil(order=6)
    Nj_Aji_star = get_recip_vectors(N, box)
    # For each atom, find the reference mesh point, and u position of the site
    m_u0, u0 = get_u_reference(positions, Nj_Aji_star)
    # find out the STGO values of each grid point
    sph_harms = sph_harmonics_GO(u0, Nj_Aji_star,shifts,n_mesh)
    # find out the local meshed values for each site
    Q_mesh_pera = Q_m_peratom(Q, sph_harms,n_mesh)
    return Q_mesh_on_m(Q_mesh_pera, m_u0, N,shifts)

def Ck_1(ksq, kappa, V):
    return 4*torch.pi/V/ksq * torch.exp(-ksq/4/kappa**2)
###########################################
def get_pme_recip(Ck_fn, kappa,positions,box,Q,K1,K2,K3, bspline_order=6):
    bspline_range = torch.arange(-bspline_order//2, bspline_order//2)
    n_mesh = (bspline_order)**3
    shifts  = make_stencil(order=6)
    # spread Q
    N = torch.tensor([K1, K2, K3])
    N_ = torch.tensor([K1, K2, K3])
    n_mesh = shifts.shape[1]
    Q_mesh = spread_Q(N,positions, box, Q,n_mesh)
    N = N.reshape((1, 1, 3))
    kpts_int = setup_kpts_integer(N)
    kpts = setup_kpts(box, kpts_int)
    half   = bspline_order // 2                     # 3 for order‑6
    m = torch.arange(-half, half).reshape(-1, 1, 1)               # p integers: −3 … +2
    theta_k = torch.prod(
            torch.sum(
                bspline(m + bspline_order/2) * torch.cos(2*torch.pi*m*kpts_int.unsqueeze(0) / N),
                axis = 0
                ),
            axis = 1
            )
    V = torch.linalg.det(box)
    #Calculate structure factor. FFT of charges on mesh grid
    S_k = torch.fft.fftn(Q_mesh).flatten()
    #Calculate fourier space convolution kernel.without k=0
    C_k = Ck_fn(kpts[3,1:], kappa, V)
    #Compute reciprocal space potential and energy. Potential is IFFT structure factor times convolution factor
    Phi_k = torch.zeros_like(S_k)
    Phi_k[1:] = C_k*S_k[1:]/torch.abs(theta_k[1:])**2
    Phi_k_3d = Phi_k.reshape(K1,K2,K3)
    Phi_real_space = torch.fft.ifftn(Phi_k_3d, norm='forward').real
    E_k = 0.5 * torch.sum(C_k * torch.abs(S_k[1:] / theta_k[1:])**2)
    #Electric field and field gradient code below
    #Default is diff_flag == 1. Use analytical differentiation of weighting function. Faster but slightly less accurate
    #Diff_flag ==2 uses ik diffentiation. Leads to most accurate force calculations. However one must do more FFTs back to real space thereby increasing complexity
    if diff_flag == 2:
        if lmax >= 1:
            kx = kpts[0].reshape(K1, K2, K3)
            ky = kpts[1].reshape(K1, K2, K3)
            kz = kpts[2].reshape(K1, K2, K3)
            # Compute E_k = i * k * Phi_k
            E_kx = 1j * kx * Phi_k_3d
            E_ky = 1j * ky * Phi_k_3d
            E_kz = 1j * kz * Phi_k_3d
            # Convert to real-space grid
            E_grid_x = torch.fft.ifftn(E_kx, norm='forward').real
            E_grid_y = torch.fft.ifftn(E_ky, norm='forward').real
            E_grid_z = torch.fft.ifftn(E_kz, norm='forward').real
            # Stack to shape [K1, K2, K3, 3]
            E_grid = torch.stack([E_grid_z, E_grid_x, E_grid_y], dim=-1)
            EG_grid = 0
        if lmax>=2:
            # Given: Phi_k_3d, kx, ky, kz of shape [K1, K2, K3]
            kx2 = kx**2
            ky2 = ky**2
            kz2 = kz**2
            kxy = kx * ky
            kxz = kx * kz
            kyz = ky * kz
            sqrt3 = torch.sqrt(torch.tensor(3.0))
            ##calculate  spherical quadrubole basis components in k-space
            Phi_k_20  = 0.5 * (2 * kz2 - kx2 - ky2) * Phi_k_3d
            Phi_k_21c = sqrt3 * kxz * Phi_k_3d
            Phi_k_21s = sqrt3 * kyz * Phi_k_3d
            Phi_k_22c = 0.5 * sqrt3* (kx2 - ky2) * Phi_k_3d
            Phi_k_22s =    sqrt3 * kxy * Phi_k_3d
            # IFFT back to real space
            EG_20  = -torch.fft.ifftn(Phi_k_20, norm='forward').real
            EG_21c = -torch.fft.ifftn(Phi_k_21c, norm='forward').real
            EG_21s = -torch.fft.ifftn(Phi_k_21s, norm='forward').real
            EG_22c = -torch.fft.ifftn(Phi_k_22c, norm='forward').real
            EG_22s = -torch.fft.ifftn(Phi_k_22s, norm='forward').real
            # Stack into real-space field gradient grid
            EG_grid = torch.stack([EG_20, EG_21c, EG_21s, EG_22c, EG_22s], dim=-1)  # shape [K1, K2, K3, 5]
        Phi_atoms, E_atoms, EG_atoms= interpolate_to_atoms(Phi_real_space,E_grid,EG_grid,positions,box,N_)
    #Else call default differeniation scheme
    else:
        E_grid, EG_grid = 0,0
        Phi_atoms, E_atoms, EG_atoms= interpolate_to_atoms(Phi_real_space, E_grid, EG_grid, positions, box, N_)
    return E_k, Phi_atoms, E_atoms ,EG_atoms
def construct_Q(q, p, t):
    """
    Constructs the full multipole moment matrix Q.
    Inputs:
        q: (N_a,) tensor of monopoles
        p: (N_a, 3) tensor of dipoles
        t: (N_a, 3, 3) tensor of quadrupoles
    Output:
        Q: (N_a, 9) tensor containing monopoles, dipoles, and quadrupoles
    """
    N_a = q.shape[0]
    # Ensure tensors are correct shape
    q = q.reshape(N_a, 1)  # (N_a, 1)
    p = p.reshape(N_a, 3)  # (N_a, 3)
    t_vec = t.reshape(N_a,5)
    # Construct full Q matrix
    Q = torch.hstack([q, p, t_vec])  # Shape: (N_a, 9)

    return Q
# ----------------------------------------------------------------------
# 1. the stencil (shifts) – used in BOTH spreading and interpolation
# ----------------------------------------------------------------------
def make_stencil(order: int):
    """
    Return the compact B‑spline stencil of length p**3 for a cardinal
    B‑spline of order p (support spans p grid intervals).

    For order = 6 ->  range = {‑3,‑2,‑1,0,1,2}   (6 points)
    For order = 5 ->  range = {‑2,‑1,0,1,2}      (5 points)
    """
    half = order // 2
    r     = torch.arange(-half, half)          # length p
    shifts = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1)
    shifts = shifts.reshape(1, order**3, 3)                   # (1, p^3, 3)
    return shifts                                             # to be reused
# ----------------------------------------------------------------------
# 2. interpolation that *re‑uses* the same stencil & theta that is used
#    in spread_Q.  lmax = 0 → only the monopole component.
# ----------------------------------------------------------------------
def interpolate_to_atoms(phi_grid: torch.Tensor,
                         E_grid: torch.Tensor,
                         EG_grid: torch.Tensor,
                         positions: torch.Tensor,
                         box: torch.Tensor,
                         N: torch.Tensor) -> torch.Tensor:
    """
    B‑spline interpolation consistent with spread_Q().
    phi_grid : (Nx,Ny,Nz) real‑space potential on the mesh
    returns   : (Natoms,) potential at atomic positions
    """
    order = 6
    # --- 1. pre‑compute quantities also used in spread_Q ----------------
    Nj_Aji_star = get_recip_vectors(N, box)                  # 3x3
    m_u0, u0    = get_u_reference(positions, Nj_Aji_star)    # (Na,3)
    shifts      = make_stencil(order)
    n_mesh      = order**3
    Na          = positions.shape[0]
    # --- 2. THETA weights with the SAME helper used in spreading ------------
    #       just call the STGO routine with lmax
    stgo = sph_harmonics_GO(u0, Nj_Aji_star, shifts, n_mesh)                            # (Na, p**3,4)
    theta = stgo[...,0]
    # --- 3. collect grid values ----------------------------------------
    Nx, Ny, Nz = N.tolist() #get mesh grid axises
    m_idx = (m_u0[:, None, :] + shifts[0]) % N[None, None, :]#compute stencil offsets around each atom in the grid and wrap them
    flat   = phi_grid.reshape(-1) #flatten grid potentials arra    
    grid_i = (m_idx[..., 0] * Ny * Nz  +
              m_idx[..., 1] * Nz +
              m_idx[..., 2])                                 # (Na, p^3)
    phi_loc = flat[grid_i]  #gather potentials from flattened grid with stencils  #(Na, p^3)
    # --- 4. weighted sum  (no renormalisation – theta already sums to 1) ---
    phi_atoms = (theta * phi_loc).sum(dim=1)                # (Na,)
    if diff_flag == 2:
        #Use ik differentiation
        #only use theta weights(aka use 6th order B-spline) for field and field gradient
        if lmax >=1:
            E_grid_flat = E_grid.reshape(-1, 3)
            E_stencil = E_grid_flat[grid_i]
            E_atoms = torch.sum(theta.unsqueeze(-1) * E_stencil, dim=1)
        if lmax>=2:
            EG_flat = EG_grid.reshape(-1, 5)
            EG_stencil = EG_flat[grid_i]
            theta /= 3.0
            EG_atoms = torch.sum(theta.unsqueeze(-1) * EG_stencil, dim=1)
    else:
        if lmax>=1:
            thetaprime_z = stgo[...,1]
            thetaprime_x = stgo[...,2]
            thetaprime_y = stgo[...,3]
            E_x = (thetaprime_x * phi_loc).sum(dim=1)  
            E_y = (thetaprime_y * phi_loc).sum(dim=1) 
            E_z = (thetaprime_z * phi_loc).sum(dim=1)
            E_atoms = torch.stack([E_z,E_x,E_y],dim=1)
        if lmax>=2:
            thetaprime2 = stgo[...,4:9] / 3 # Must divide by (2l-1)!!=3 for l=2. Normalization for quadrupoles
            thetaprime2_1 = thetaprime2[...,0]  
            thetaprime2_2 = thetaprime2[...,1]  
            thetaprime2_3 = thetaprime2[...,2]  
            thetaprime2_4 = thetaprime2[...,3] 
            thetaprime2_5 = thetaprime2[...,4]
            EG_1 = (thetaprime2_1 * phi_loc).sum(dim=1)
            EG_2 = (thetaprime2_2 * phi_loc).sum(dim=1)
            EG_3 = (thetaprime2_3 * phi_loc).sum(dim=1)
            EG_4 = (thetaprime2_4 * phi_loc).sum(dim=1)
            EG_5 = (thetaprime2_5 * phi_loc).sum(dim=1)
            EG_atoms = torch.stack([EG_1,EG_2,EG_3,EG_4,EG_5], dim=1)
    return phi_atoms, E_atoms, EG_atoms


def compute_pme(coords, q ,p, t,box, rcutoff, thresh):
#Total energy = short range + long range - self interaction
  #kappa, K1, K2, K3 = setup_ewald_parameters(rcutoff, thresh,box)
  kappa = 0.544590516336201*BOHR2ANG
  K1,K2,K3 = 24,24,24
  print("kappa: ", kappa)
  print("K1: ", K1)
  U_s = 0 #short_range(coords, q,p,t, box, kappa, rcutoff) 
  U_self_vectorized = 0# self_interaction_vectorized(coords,q,p,t,kappa)
  Q = construct_Q(q,p,t)
  result = get_pme_recip(Ck_1, kappa,coords,box,Q,K1,K2,K3,bspline_order=6)
  U_l = result[0]
  V_l = result[1]
  E_l = result[2]
  E_g_l = result[3]
  U_ewald =  U_l + U_s - U_self_vectorized
  print(f"FINAL ENERGIES(HARTREE): U_s = {U_s} , U_l = {U_l} , U_self = {U_self_vectorized}, Total Coulombic Energy = {U_ewald}")
  print(f"FINAL ENERGIES(KCAL/MOL): U_s = {U_s*HARTREE2KCAL} , U_l = {U_l*HARTREE2KCAL} , U_self = {U_self_vectorized*HARTREE2KCAL}, Total Coulombic Energy = {U_ewald*HARTREE2KCAL}")
  return U_ewald, V_l, E_l, E_g_l

