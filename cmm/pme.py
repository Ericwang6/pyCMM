import torch
from scipy import special
from torch.special import erfc
import math
from . import units, pbc
from .units import EPSILON0 , ELE_CHG 
from .pbc import applyPBC
import numpy as np
torch.set_printoptions(profile="full")
lmax = 0
# This file contains the Ewald Summation for computing Long range interactions
def short_range(coords, q , p, t, box, kappa, rcutoff):
  U_cc, U_cd, U_dd, U_ct, U_dt, U_tt = 0, 0, 0, 0, 0, 0
  N = len(coords) #atoms
  boxInv = torch.linalg.inv(box)  # Inverse of the simulation box matrix
  for i in range(N):
    for j in range(i+1,N):
      drVec = coords[j] - coords[i]
      r_ = applyPBC(drVec.unsqueeze(0), box, boxInv).squeeze(0)
      r = torch.norm(r_) #Get distance between two points.
      if r < rcutoff:
#        r2 = r**2
#        r3 =r2*r
#        r4 = r3*r
#        r5 = r3*r2
#        r7 = r5*r2
#        r9 = r7*r2
#        kappa2 = kappa*kappa
#        kappa3 = kappa2*kappa
#        delta = torch.eye(3) #kronecker delta
        ########|||||Ewald screening functions are represented by f_ |||||#######
#        f_0 = erfc(r * kappa) 
#        f_1 = (2*kappa*r/math.pi * torch.exp(-kappa2 * r2)) + f_0
#        f_2 = 4 * kappa3/math.sqrt(math.pi) * torch.exp(-kappa2 * r2)/r2
#        f_3 = r3 * f_2
#        f_4 = 8 * kappa/math.pi * (kappa2 * r2 + 1)/r4 * torch.exp(-kappa2 * r2)
#        f_5 = r3 * f_4 - 3 * r * f_2
#        f_6 = 2*(kappa2 + 1/r2) * f_4 + 4*f_2/r4
        #######|||||Multipole tensors are represented by T_. Also implemented are the D_ tensors|||||#######
        T_ = 1/r
#        T_a = -r_/r3
#        T_ab = (3 * torch.outer(r_,r_) - r2 * delta)/r5
        #################################################################
#        term1_abg = 15 * torch.einsum("i,j,k->ijk", r_,r_,r_)
#        term2_abg = -3 * r2 *(
#                    torch.einsum("i,jk->ijk", r_, delta)+
#                    torch.einsum("j,ik->ijk", r_, delta)+
#                    torch.einsum("k,ij->ijk", r_, delta)
#        )
#        T_abg = -(term1_abg + term2_abg) / r7
        #################################################################
#        term1_abgd = 105 * torch.einsum("i,j,k,l -> ijkl", r_,r_,r_,r_)
#        term1 = torch.einsum("i,j,kl->ijkl", r_, r_, delta)  # R_ij,a R_ij,b k_gd
#        term2 = torch.einsum("i,k,jl->ijkl", r_, r_, delta)  # R_ij,a R_ij,g k_bd
#        term3 = torch.einsum("i,l,jk->ijkl", r_, r_, delta)  # R_ij,a R_ij,d k_bg
#        term4 = torch.einsum("j,k,il->ijkl", r_, r_, delta)  # R_ij,b R_ij,g k_ad
#        term5 = torch.einsum("j,l,ik->ijkl", r_, r_, delta)  # R_ij,b R_ij,d k_ag
#        term6 = torch.einsum("k,l,ij->ijkl", r_, r_, delta)  # R_ij,g R_ij,d k_ab
#        term2_abgd = -15 * r2 * (term1 + term2 + term3 + term4 + term5 + term6)
#        term3_abgd = 3 * r4 * (
#          torch.einsum("ij,kl->ijkl", delta, delta) +
#          torch.einsum("ik,jl->ijkl", delta, delta) +
#          torch.einsum("il,jk->ijkl", delta, delta)
#        )
#        T_abgd = (term1_abgd +term2_abgd + term3_abgd)/r9 #final term
#        D_ab = torch.einsum("i,j->ij", r_,r_)
#        D_abgd = torch.einsum("ij,k,l -> ijkl", delta, r_, r_) + torch.einsum("ik,j,l->ijkl", delta, r_, r_) + torch.einsum("il,j,k->ijkl", delta, r_, r_) 
        ###charge-charge###
#        U_cc += (q[i] * q[j])*f_0*T_    
        ###charge-dipole###
#        U_cd += -f_1*(q[i]*torch.dot(T_a,p[j]) - q[j]*torch.dot(T_a,p[i]))
        ###dipole-dipole###
#        U_dd += -(f_1* torch.dot(p[i],torch.matmul(T_ab,p[j])) + f_2 *torch.dot(p[i],torch.matmul(D_ab,p[j])))
        ###charge-quadrupole###
#        U_ct += f_1*(q[i]*torch.sum(T_ab * t[j]) + q[j]* torch.sum(T_ab*t[i]))  + f_2*(q[i] * torch.sum(D_ab * t[j]) + q[j] * torch.sum(D_ab * t[i]))
        ###dipole-quadrupole###
#        term1_dt = f_1 * (torch.einsum("i,ijk,ik->",p[i], T_abg,t[j]) - torch.einsum("i,ijk,ik->",p[j], T_abg,t[i   ]))
#        term2_dt = f_2 * (torch.einsum("i,j,ij->", p[i],r_,t[j]) - torch.einsum("i,j,ij->", p[j],r_,t[i]))
#        term3_dt = -f_3 * (torch.einsum("i,i,jk,jk->",p[i],r_,T_ab,t[j]) - torch.einsum("i,i,jk,jk->",p[j],r_,T_ab,t[i]))
#        term4_dt = -f_4 * (torch.einsum("i,i,jk,jk->",p[i],r_,D_ab,t[j]) - torch.einsum("i,i,jk,jk->",p[j],r_,D_ab,t[i])) 
#        U_dt += term1_dt + term2_dt + term3_dt + term4_dt
        ###quadrupole-quadrupole###
#        term1_tt = f_1 * torch.einsum("ij,ijkl,kl->",t[i],T_abgd,t[j])
#        term2_tt = 2 * f_2 * torch.einsum("ij,ij->",t[i],t[j])
#        term3_tt = f_3 * (torch.einsum("ij,ij,kl,kl->",t[i],delta,T_ab,t[j]) + torch.einsum("ij,ij,kl,kl",t[i],T_ab,delta,t[j]))/2
#        term4_tt = -f_3 * (torch.einsum("ij,i,jkl,kl->",t[i],r_,T_abg,t[j]) + torch.einsum("ij,i,jkl,kl->",t[j],r_,T_abg,t[i]))
#        term5_tt = -f_4 * (torch.einsum("ij,ijkl,kl->",t[i],D_abgd,t[j]) + torch.einsum("ij,klij,kl->",t[i],D_abgd,t[j]))/2 
#        term6_tt = -2 * f_4 * torch.einsum("i,ij,jk,k->",r_,t[i],t[j],r_)
#        term7_tt = f_5 * 1/2 * (torch.einsum("ij,ij->",D_ab,t[i]) * torch.einsum("ij,ij->", T_ab,t[j])+
#                   torch.einsum("ij,ij->",D_ab,t[j]) * torch.einsum("ij,ij->", T_ab,t[i])
#                   ) 
#        term8_tt = f_6 * torch.einsum("ij,ij->",D_ab,t[i]) * torch.einsum("ij,ij->",D_ab,t[j]) 
#        U_tt = term1_tt + term2_tt + term3_tt + term4_tt + term5_tt + term6_tt + term7_tt + term8_tt
#  U_ct *= 1/3
#  U_dt *= 1/3
#  U_tt *= 1/9
  U_s = U_cc# + U_cd + U_dd + U_ct + U_dt + U_tt
  return U_s
    
def self_interaction_vectorized(coords,q,p,t,kappa):
#Vectorized version of self interaction function
    U_q = torch.dot(q,q)
#    U_d = torch.sum(p*p)
#    U_cq = torch.dot(q,torch.einsum("bii->b",t))
#    U_t = torch.sum(t*t)
    #multiplying with correct constants
    U_q *= kappa/math.sqrt(torch.pi)
#    U_d *= 2 * kappa**3 /(3 * math.sqrt(torch.pi))
#    U_cq *= 2*kappa**3/(3 * torch.pi)
#    U_t *= 8*kappa**5/(45*torch.pi)
    #total self-interaction energy
    U_self = U_q# + U_d + U_cq + U_t
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
    #print("COORDS SHAPE: ", coords.shape)
    #print("NJAJISTAR SHAPE: ", Nj_Aji_star.shape)
    R_in_m_basis = torch.einsum("ij,kj->ki",Nj_Aji_star,coords)
    m_u0 = torch.ceil(R_in_m_basis).to(torch.int32)
    u0 = (m_u0 - R_in_m_basis) + bspline_order/2
    return m_u0, u0

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
    div_00 = torch.tensor(M2prime_u[:, 0] * M_u[:, 1] * M_u[:, 2])
    div_11 = torch.tensor(M2prime_u[:, 1] * M_u[:, 0] * M_u[:, 2])
    div_22 = torch.tensor(M2prime_u[:, 2] * M_u[:, 0] * M_u[:, 1])
    div_01 = torch.tensor(Mprime_u[:, 0] * Mprime_u[:, 1] * M_u[:, 2])
    div_02 = torch.tensor(Mprime_u[:, 0] * Mprime_u[:, 2] * M_u[:, 1])
    div_12 = torch.tensor(Mprime_u[:, 1] * Mprime_u[:, 2] * M_u[:, 0])
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

    Inputs:
        u0: 
            a N_a * 3 matrix containing all positions
        Nj_Aji_star:
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
    #u = (u0[:, torch.unsqueeze(0), :] + shifts).reshape((N_a*n_mesh, 3)) 
    #print("U0 SHAPE")
    #print(u0.shape)
    #print("SHIFTS SHAPE")
    #print(shifts.shape)
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
    Computes <R_t|Q>. See eq. (49) of https://doi.org/10.1021/ct5007983
    
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
        Q_dbf = torch.hstack([Q_dbf, Q[:,4:9]/3])
    #Q_m_pera = torch.sum(Q_dbf[:,torch.unsqueeze(0),:]* sph_harms, axis=2)                                          
    Q_m_pera = torch.sum(Q_dbf.unsqueeze(1) * sph_harms, dim=2)
    assert Q_m_pera.shape == (N_a, n_mesh)
    return Q_m_pera
def Q_mesh_on_m(Q_mesh_pera, m_u0, N,shifts):
    """
    Reduce the local Q_m_peratom into the global mesh
    
    Input:
        Q_mesh_pera, m_u0, N
        
    Output:
        Q_mesh: 
            Nx * Ny * Nz matrix
    """
    indices_arr = np.mod(m_u0[:,np.newaxis,:]+shifts, N[np.newaxis, np.newaxis, :])
    ### jax trick implementation without using for loop
    ### NOTICE: this implementation does not work with numpy!
    Q_mesh = torch.zeros((N[0], N[1], N[2]))
    #Q_mesh = Q_mesh.at[indices_arr[:, :, 0], indices_arr[:, :, 1], indices_arr[:, :, 2]].add(Q_mesh_pera)
    #Q_mesh.scatter_add_(0, indices_arr[:, :, 0], Q_mesh_pera)
    Q_mesh.index_put_((indices_arr[:, :, 0], indices_arr[:, :, 1], indices_arr[:, :, 2]), Q_mesh_pera, accumulate=True)

    return Q_mesh
def setup_kpts_integer(N):
    """
    Outputs:
        kpts_int:
            n_k * 3 matrix, n_k = N[0] * N[1] * N[2]
    """
    #N_half = N.reshape(3)
    N_half = N.reshape(3).tolist()
    #kx, ky, kz = [torch.roll(torch.arange(- (N_half[i] - 1) // 2, (N_half[i] + 1) // 2 ), - (N_half[i] - 1) // 2) for i in range(3)]
    kx, ky, kz = [torch.roll(torch.arange(- (int(N_half[i]) - 1) // 2, (int(N_half[i]) + 1) // 2), shifts=- (int(N_half[i]) - 1) // 2) for i in range(3)]
    # kpts_int = jnp.hstack([ki.flatten()[:,jnp.newaxis] for ki in jnp.meshgrid(kz, kx, ky)])
    kpts_int = torch.hstack([ki.flatten().unsqueeze(1) for ki in torch.meshgrid(kx, ky, kz, indexing='ij')])
    #kpts_int = kpts_int.to(torch.float32)
    #print("KPOINTS_INT SHAPE(SHOULD BE n_k x 3): ", kpts_int.shape)
    #print("KPOINTS_INT: ", kpts_int)
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
    # 4 * K
    #kpts = torch.hstack((kpts, ksq[:, torch.unsqueeze(0)])).T
    kpts = torch.hstack((kpts, ksq.unsqueeze(1))).T
    return kpts
def spread_Q(N,positions, box, Q,shifts,n_mesh):
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
    Nj_Aji_star = get_recip_vectors(N, box)
    # For each atom, find the reference mesh point, and u position of the site
    m_u0, u0 = get_u_reference(positions, Nj_Aji_star)
    # find out the STGO values of each grid point
    sph_harms = sph_harmonics_GO(u0, Nj_Aji_star,shifts,n_mesh)
    # find out the local meshed values for each site
    Q_mesh_pera = Q_m_peratom(Q, sph_harms,n_mesh)
    return Q_mesh_on_m(Q_mesh_pera, m_u0, N,shifts)
def Ck_1(ksq, kappa, V):
    return 2*torch.pi/V/ksq * torch.exp(-ksq/4/kappa**2)

def Ck_6(ksq, kappa, V):
    thresh = 1e-16
    #ksq = np.piecewise(ksq, [ksq<thresh, ksq>=thresh], [lambda x: np.array(thresh), lambda x: x])
    sqrt_pi = math.sqrt(torch.pi)
    ksq = torch.where(ksq < thresh, torch.tensor(thresh, dtype=torch.float32), ksq)
    x2 = ksq / 4 / kappa**2
    x = torch.sqrt(x2)
    x3 = x2 * x
    exp_x2 = torch.exp(-x2)
    f = (1 - 2*x2)*exp_x2 + 2*x3*sqrt_pi*erfc(x)
    return sqrt_pi*torch.pi/2/V*kappa**3 * f / 3

def Ck_8(ksq, kappa, V):
    thresh = 1e-16
    #ksq = np.piecewise(ksq, [ksq<thresh, ksq>=thresh], [lambda x: np.array(thresh), lambda x: x])
    sqrt_pi = math.sqrt(torch.pi)
    ksq = torch.where(ksq < thresh, torch.tensor(thresh, dtype=torch.float32), ksq)
    x2 = ksq / 4 / kappa**2
    x = torch.sqrt(x2)
    x4 = x2 * x2
    x5 = x4 * x
    exp_x2 = torch.exp(-x2)
    f = (3 - 2*x2 + 4*x4)*exp_x2 - 4*x5*sqrt_pi*erfc(x)
    return sqrt_pi*torch.pi/2/V*kappa**5 * f / 45

def Ck_10(ksq, kappa, V):
    thresh = 1e-16
    #ksq = np.piecewise(ksq, [ksq<thresh, ksq>=thresh], [lambda x: np.array(thresh), lambda x: x])
    sqrt_pi = math.sqrt(torch.pi)
    ksq = torch.where(ksq < thresh, torch.tensor(thresh, dtype=torch.float32), ksq)
    x2 = ksq / 4 / kappa**2
    x = torch.sqrt(x2)
    x4 = x2 * x2
    x6 = x4 * x2
    x7 = x6 * x
    exp_x2 = torch.exp(-x2)
    f = (15 - 6*x2 + 4*x4 - 8*x6)*exp_x2 + 8*x7*sqrt_pi*erfc(x)
    return sqrt_pi*torch.pi/2/V*kappa**7 * f / 1260
def interpolate_to_atoms(Phi_real_space, positions, box, N):
    Nj_Aji_star = get_recip_vectors(N, box)
    m_u0, u0 = get_u_reference(positions, Nj_Aji_star)
    # Compute B-spline weights and derivatives
    M_u = bspline(u0)
    Mprime_u = bspline_prime(u0)
    M2prime_u = bspline_prime2(u0)
    #print("BSPLINE: ",M_u)
    #print("BSPLINE PRIME: ",Mprime_u)
    #print("BSPLINE PRIME2: ",M2prime_u)
    # Compute theta, thetaprime, and theta2prime
    theta = get_theta(u0, M_u)  # Monopole interpolation
    thetaprime = get_thetaprime(u0, Nj_Aji_star, M_u, Mprime_u)  # Dipole interpolation
    theta2prime = get_theta2prime(u0, Nj_Aji_star, M_u, Mprime_u, M2prime_u)  # Quadrupole interpolation
    #print("THETA: ", theta)
    #print("THETAPRIME: ", thetaprime)
    #print("THETA2PRIME: ", theta2prime)
    # Debugging
    #print("Phi_real_space shape:", Phi_real_space.shape)
    #print("m_u0 shape:", m_u0.shape)
    #print("m_u0 min/max:", m_u0.min(), m_u0.max())
    m_u0 = torch.remainder(m_u0, torch.tensor([N[0], N[1], N[2]]))
    #print("m_u0: ", m_u0)
    #print("m_u0 SHAPE: ", m_u0.shape)
    # Sum the contributions from neighboring grid points
    Phi_real_space = Phi_real_space.contiguous()
    Phi_atoms = torch.zeros((m_u0.shape[0],), dtype=Phi_real_space.dtype)
    Phi_dipole = torch.zeros_like(Phi_atoms)
    Phi_quadrupole = torch.zeros_like(Phi_atoms)
    for i in range(m_u0.shape[0]):  # Loop over each atom
        # Extract potential from grid using wrapped indices
        Phi_grid = Phi_real_space[m_u0[i, 0], m_u0[i, 1], m_u0[i, 2]]
        # Monopole contribution
        Phi_atoms[i] = torch.sum(theta[i] * Phi_grid)
        # Dipole contribution (dot product with gradient)
        Phi_dipole[i] = torch.sum(thetaprime[i] * Phi_grid)
        # Quadrupole contribution (double contraction with Hessian)
        Phi_quadrupole[i] = torch.sum(theta2prime[i] * Phi_grid)
    print("MONOPOLE RECIPROCAL POTENTIAL CONTRIBUTION: ", Phi_atoms)
    print("DIPOLE RECIPROCAL POTENTIAL CONTRIBUTION: ", Phi_dipole)
    print("QUADRUPOLE RECIPROCAL POTENTIAL CONTRIBUTION: ", Phi_quadrupole)
    Phi_tot = Phi_atoms + Phi_dipole + Phi_quadrupole
    return Phi_tot
###########################################
def get_pme_recip(Ck_fn, kappa,positions,box,Q, bspline_order=6,K1=32,K2=32,K3=32,lmax=2):
    bspline_range = torch.arange(-bspline_order//2, bspline_order//2)
    #print("BSPLINE_RANGE: ", bspline_range)
    n_mesh = bspline_order**3
    shifts = torch.stack(torch.meshgrid(bspline_range, bspline_range, bspline_range)).T.reshape((1, n_mesh, 3))
    #print("SHIFTS: ", shifts)
    # spread Q
    N_= torch.tensor([K1,K2,K3]) #For potential calculations
    N = torch.tensor([K1, K2, K3])
    Q_mesh = spread_Q(N,positions, box, Q,shifts,n_mesh)
    #print("QMESH: ", Q_mesh)
    N = N.reshape((1, 1, 3))
    kpts_int = setup_kpts_integer(N)
    kpts = setup_kpts(box, kpts_int)
   # print("KPOINTS: ", kpts)
    m = torch.linspace(-bspline_order//2+1, bspline_order//2-1, bspline_order-1).reshape(bspline_order-1, 1, 1)
    theta_k = torch.prod(
            torch.sum(
                bspline(m + bspline_order/2) * torch.cos(2*torch.pi*m*kpts_int.unsqueeze(0) / N),
                axis = 0
                ),
            axis = 1
            )
    #print("THETA_K: ", theta_k)
    V = torch.linalg.det(box)
    #Calculate structure factor. 3d form is used for potential. Flattened form for energy
    S_k_3d = torch.fft.fftn(Q_mesh)
    #print("S_K_3d SHAPE: ", S_k_3d.shape)
    S_k = S_k_3d.flatten()
    #print("S_K SHAPE: ", S_k.shape)
    #print("STRUCTURE FACTOR: ", S_k)

    #Calculate fourier space convolution kernel.include k=0
    C_k_full = Ck_fn(kpts[3], kappa, V)
    #print("C_k SHAPE: ", C_k_full.shape)
    C_k_3d = C_k_full.reshape((K1, K2, K3))  # Reshape to match grid
    #print("C_k_3d SHAPE: ", C_k_3d.shape)
    # Ensure `theta_k` is reshaped to match the grid
    theta_k_3d = theta_k.reshape((K1, K2, K3))
    #Compute reciprocal space potential
    Phi_k = torch.zeros_like(S_k_3d)  # Ensure same size as S_k_3d
    Phi_k[1:] = C_k_3d[1:] * S_k_3d[1:] / theta_k_3d[1:]
    #Inverse FFT to get long-range potential on grid
    Phi_real_space = torch.fft.ifftn(Phi_k).real
    #print("Phi_REAL_SPACE: ", Phi_real_space)
    #print("Phi_REAL_SPACE SHAPE: ", Phi_real_space.shape)
    #Interpolate long-range potential back to atomic positions
    Phi_atoms = interpolate_to_atoms(Phi_real_space,positions,box,N_)
    
    #Back to computing energy
    C_k = C_k_full[1:]
    E_k = C_k * torch.abs(S_k[1:] / theta_k[1:])**2
    E_k = torch.sum(E_k) * EPSILON0
#    print(f"IFFT: {Phi_real_space} IFFT shape: {Phi_real_space.shape}")

    return E_k, Phi_atoms
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
    # Convert quadrupoles to a 5-element vector
    t_vec = torch.stack([
        t[:, 0, 0],  # Q_xx
        t[:, 1, 1],  # Q_yy
        t[:, 2, 2],  # Q_zz
        t[:, 0, 1],  # Q_xy
        t[:, 0, 2],  # Q_xz
        #t[:, 1, 2],  # Q_yz
    ], dim=1)

    # Construct full Q matrix
    Q = torch.hstack([q, p, t_vec])  # Shape: (N_a, 9)

    return Q

def compute_pme(coords, q ,p, t,box, kappa, rcutoff, kcutoff):
#Total energy = short range + long range - self interaction
  #print("CHARGES: ", q)
  #print("DIPOLES: ", p)
  #print("QUADRUPOLES: ", t)
  U_s = short_range(coords, q,p,t, box, kappa, rcutoff) 
  U_self_vectorized = self_interaction_vectorized(coords,q,p,t,kappa)
  Q = construct_Q(q,p,t)
  #print("FULL MULTIPOLE MOMENT MATRIX: ", Q)
  #print("Q SHAPE: ", Q.shape)
  #C_k I believe is the fourier convolution factor <k|x>(4pi/k**2) in the paper
  #C_k_10 accounts for all multipoles
  #Ck_list = [Ck_1, Ck_6, Ck_8, Ck_10]
  Ck_list = [Ck_1]
  for Ck in Ck_list:
    result = get_pme_recip(Ck, kappa,coords,box,Q,bspline_order=6,K1=15,K2=15,K3=15,lmax=0)
    U_l = result[0]
    V_l = result[1]
#    print("LONG RANGE POTENTIAL: ", V_l)
  #U_s, U_l = 0,0
 # U_self = self_interaction(coords, q,p,t, kappa)
    U_ewald =  U_l + U_s - U_self_vectorized
    print(f"USING CK: {Ck}")
    print(f"FINAL ENERGIES: U_s = {U_s} , U_l = {U_l} , U_self = {U_self_vectorized}, Total Coulombic Energy = {U_ewald}")
  return U_ewald

