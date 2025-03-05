import torch
from scipy import special
from torch.special import erfc
import math
from . import units, pbc
from .units import EPSILON0 , ELE_CHG 
from .pbc import applyPBC
import numpy as np
torch.set_printoptions(profile="full")
#Calculate electric field and charge potential
def get_real_space_field(coords, q , p, t, box, kappa):
  N = len(coords)
  E_real = torch.zeros((N, 3)) 
  boxInv = torch.linalg.inv(box)  # Inverse of the simulation box matrix
  for i in range(N):
    for j in range(N):
      if i==j: continue
      drVec = coords[j] - coords[i]
      r_ = applyPBC(drVec.unsqueeze(0), box, boxInv).squeeze(0)
      r = torch.norm(r_) #Get distance between two points.
      rhat = r_ / r
      r2 = r**2
      r3 =r2*r
      r4 = r3*r
      r5 = r3*r2
      r7 = r5*r2
      r9 = r7*r2
      kappa2 = kappa*kappa
      kappa3 = kappa2*kappa
      delta = torch.eye(3) #kronecker delta
      #######||||| Screened Multipole tensors are represented by sT_. screened interaction tensors are represetned by sT_|||||#######
      pikappa = math.sqrt(math.pi)*kappa
      r2_inverse = 1/r2
      expkappa = torch.exp(-kappa2*r2)
      screened_r3 = r2_inverse * (1/r + 2*kappa2* expkappa/pikappa)
      screened_r5 = r2_inverse * (1/r3 + (2*kappa2)**2 * expkappa/(3*pikappa))
      screened_r7 = r2_inverse * (1/r5 + (2*kappa2)**3 * expkappa/(5*pikappa))
      screened_r9 = r2_inverse * (1/r7 + (2*kappa2)**4 * expkappa/(7*pikappa))
      sT_a = screened_r3 * -r_ 
      sT_ab = screened_r5 * (3 * torch.outer(r_,r_) - r2 * delta) 
      
      #################################################################
      term1_abg = 15 * torch.einsum("i,j,k->ijk", r_,r_,r_)
      term2_abg = -3 * r2 *(
                torch.einsum("i,jk->ijk", r_, delta)+
                torch.einsum("j,ik->ijk", r_, delta)+
                torch.einsum("k,ij->ijk", r_, delta)
      )
      sT_abg = -screened_r7*(term1_abg + term2_abg)
      #################################################################
      E_real[i] += -q[j] * sT_a + torch.einsum("j,ij->i",p[j],sT_ab) -torch.einsum("jk,ijk->i",t[j],sT_abg)
  return E_real
def get_reciprocal_space_field(coords, q, p , t ,box, kappa,kcutoff,kvectors):
  V = torch.abs(torch.dot(box[2],torch.linalg.cross(box[0],box[1]))) #compute volume of box
  constant = 8*math.pi/V
  #Precalculating gaussian factors
  k_squared = torch.einsum("ij,ij->i", kvectors, kvectors)  # Shape: (N_k,)
  gaussian_factor = torch.exp(-k_squared /(4*kappa**2)) / k_squared
  #Initialize all structure factors
  sk_cos = torch.zeros(len(kvectors))
  sk_sin = torch.zeros(len(kvectors))
  sk_dipole_cos = torch.zeros(len(kvectors))
  sk_dipole_sin = torch.zeros(len(kvectors))
  sk_quad_cos = torch.zeros(len(kvectors))
  sk_quad_sin = torch.zeros(len(kvectors))
  N = len(coords)
  E_c, E_d, E_t = torch.empty(N,3), torch.empty(N,3), torch.empty(N,3)
  #Compute all structure factors
  for i, kvec in enumerate(kvectors):
    k_dot_r = torch.sum(kvec * coords, dim=1)
    k_dot_p = torch.sum(p * kvec, dim=1)
    k_outer = torch.outer(kvec, kvec)         # h_a h_b
    h_contract_theta = torch.sum(k_outer * t, dim = (1,2))  # h_a h_b theta_iab
    sk_cos[i] = torch.sum(q * torch.cos(k_dot_r))
    sk_sin[i] = torch.sum(q * torch.sin(k_dot_r))
    sk_dipole_cos[i] = torch.sum(k_dot_p * torch.cos(k_dot_r))
    sk_dipole_sin[i] = torch.sum(k_dot_p * torch.sin(k_dot_r))
    sk_quad_cos[i] = torch.sum(h_contract_theta * torch.cos(k_dot_r))
    sk_quad_sin[i] = torch.sum(h_contract_theta * torch.sin(k_dot_r))
  #Compute sin and cosine terms for unique particles. Then compute their electric fields
  for j in range(len(coords)):
    sin_term = torch.sum(torch.sin(torch.einsum("ki,i->k", kvectors, coords[j])))  # Sum over k
    cos_term = torch.sum(torch.cos(torch.einsum("ki,i->k", kvectors, coords[j]))) 
    E_c[j] = torch.sum(gaussian_factor[:, None] * kvectors * (cos_term * sk_sin.sum() - sin_term * sk_cos.sum()), dim=0)
    #E_c[j] = torch.sum(gaussian_factor * kvectors*(cos_term*torch.sum(sk_sin) - sin_term*torch.sum(sk_cos)))
    #E_d[j] = torch.sum(gaussian_factor * kvectors*(cos_term*torch.sum(sk_dipole_cos) + sin_term*torch.sum(sk_dipole_sin)))
    #E_t[j] = torch.sum(gaussian_factor * kvectors*(sin_term*torch.sum(sk_quad_cos) - cos_term*torch.sum(sk_quad_sin)))
    E_d[j] = torch.sum(gaussian_factor[:,None] * kvectors*(cos_term*sk_dipole_cos.sum() + sin_term*sk_dipole_sin.sum()),dim=0)
    E_t[j] = torch.sum(gaussian_factor[:,None] * kvectors*(sin_term*sk_quad_cos.sum() - cos_term*sk_quad_sin.sum()),dim=0)
  E_c *= -constant
  E_d *= constant
  E_t *= -constant/3
  print("REC FIELD DUE TO CHARGES: ", E_c)
  print("REC FIELD DUE TO DIPOLES: ", E_d)
  print("REC FIELD DUE TO QUADRUPOLES: ", E_t)
  E_recip = E_c + E_d + E_t
  return E_recip

def get_self_interaction_field_vectorized(coords,q,p,t,box,kappa):
    E_self = p * (-4*kappa**3)/(3*math.sqrt(torch.pi))
    return E_self
def get_real_space_potential(coords,q,p,t,box,kappa):
  N = len(coords)
  P_qq, P_qp, P_qt = torch.zeros(N), torch.zeros(N), torch.zeros(N) 
  kappa2 = kappa*kappa
  kappa3 = kappa2*kappa
  delta = torch.eye(3)
  boxInv = torch.linalg.inv(box)  # Inverse of the simulation box matrix
  for i in range(N):
    P_qq_i , P_qp_i, P_qt_i = 0,0,0
    for j in range(N):
      if i==j: continue
      drVec = coords[j] - coords[i]
      r_ = applyPBC(drVec.unsqueeze(0), box, boxInv).squeeze(0)
      r = torch.norm(r_) #Get distance between two points. 
      r2 = r*r
      r3 = r2*r
      r5 = r3*r2
      f_0 = erfc(r * kappa) 
      f_1 = (2*kappa*r/math.pi * torch.exp(-kappa2 * r2)) + f_0
      f_2 = 4 * kappa3/math.sqrt(math.pi) * torch.exp(-kappa2 * r2)/r2
      T_ = 1/r
      T_a = -r_/r3
      T_ab = (3 * torch.outer(r_,r_) - r2 * delta)/r5
      D_ab = torch.einsum("i,j->ij", r_,r_)
      #---------------------------------------------------------------#
      P_qq_i += f_0*T_*q[j]
      P_qp_i += f_1*torch.einsum("i,i->",T_a,p[j])
      P_qt_i += f_1*torch.einsum("ij,kl->",T_ab,t[j]) + f_2*torch.einsum("ij,kl->",D_ab,t[j])
    P_qq[i] = P_qq_i
    P_qp[i] = -P_qp_i
    P_qt[i] = P_qt_i/3
  P_real = P_qq + P_qp + P_qt
  return P_real
def get_reciprocal_space_potential(coords, q, p , t ,box, kappa,kcutoff,kvectors):
  V = torch.abs(torch.dot(box[2],torch.linalg.cross(box[0],box[1]))) #compute volume of box
  constant = 8*math.pi/V
  #Precalculating gaussian factors
  k_squared = torch.einsum("ij,ij->i", kvectors, kvectors)  # Shape: (N_k,)
  print("NUMBER OF KVECTORS: ", len(kvectors))
  gaussian_factor = torch.exp(-k_squared /(4*kappa**2)) / k_squared
  #Initialize all structure factors
  sk_cos = torch.zeros(len(kvectors))
  sk_sin = torch.zeros(len(kvectors))
  sk_dipole_cos = torch.zeros(len(kvectors))
  sk_dipole_sin = torch.zeros(len(kvectors))
  sk_quad_cos = torch.zeros(len(kvectors))
  sk_quad_sin = torch.zeros(len(kvectors))
  N = len(coords)
  P_qq,P_qu,P_qt = torch.empty(N), torch.empty(N), torch.empty(N)
  #Compute all structure factors
  for i, kvec in enumerate(kvectors):
    k_dot_r = torch.sum(kvec * coords, dim=1)
    k_dot_p = torch.sum(p * kvec, dim=1)
    k_outer = torch.outer(kvec, kvec)         # h_a h_b
    h_contract_theta = torch.sum(k_outer * t, dim = (1,2))  # h_a h_b theta_iab
    sk_cos[i] = torch.sum(q * torch.cos(k_dot_r))
    sk_sin[i] = torch.sum(q * torch.sin(k_dot_r))
    sk_dipole_cos[i] = torch.sum(k_dot_p * torch.cos(k_dot_r))
    sk_dipole_sin[i] = torch.sum(k_dot_p * torch.sin(k_dot_r))
    sk_quad_cos[i] = torch.sum(h_contract_theta * torch.cos(k_dot_r))
    sk_quad_sin[i] = torch.sum(h_contract_theta * torch.sin(k_dot_r))
  #Compute sin and cosine terms for unique particles. Then compute their potentials
  for j in range(len(coords)):
    sin_term = torch.sum(torch.sin(torch.einsum("ki,i->k", kvectors, coords[j])))  # Sum over k
    cos_term = torch.sum(torch.cos(torch.einsum("ki,i->k", kvectors, coords[j]))) 
    P_qq[j] = torch.sum(gaussian_factor * (cos_term*torch.sum(sk_cos) +sin_term*torch.sum(sk_sin)))
    P_qu[j] = torch.sum(gaussian_factor * (sin_term*torch.sum(sk_dipole_cos)-cos_term*torch.sum(sk_dipole_sin)))
    P_qt[j] = torch.sum(gaussian_factor * (cos_term*torch.sum(sk_quad_cos) +sin_term*torch.sum(sk_quad_sin)))
  P_qq *= constant
  P_qu *= constant
  P_qt*= -constant/3
  print("REC CHARGE-CHARGE POTENTIAL: ", P_qq)
  print("REC CHARGE-DIPOLE POTENTIAL: \n", P_qu)
  print("REC CHARGE-QUADRUPOLE POTENTIAL: \n", P_qt)
  P_recip = P_qq + P_qu + P_qt
  return P_recip
def get_reciprocal_space_potential_vectorized(coords, q, p, t, box, kappa, kcutoff, kvectors):
    #WARNING THIS IS NOT CORRECT. NEED TO FINISH
    """
    Compute the reciprocal space potential in PME.

    Inputs:
        coords: (N,3) tensor - Atomic coordinates.
        q: (N,) tensor - Monopoles (charges).
        p: (N,3) tensor - Dipoles.
        t: (N,3,3) tensor - Quadrupoles.
        box: (3,3) tensor - Simulation box.
        kappa: float - Ewald parameter.
        kcutoff: float - Reciprocal space cutoff.
        kvectors: (N_k,3) tensor - Reciprocal lattice vectors.

    Output:
        P_recip: (N,) tensor - The reciprocal space potential at each atomic site.
    """
    # Compute volume of the box
    V = torch.abs(torch.dot(box[2], torch.linalg.cross(box[0], box[1])))
    constant = (8 * math.pi) / V
    # Compute squared k-values and Gaussian damping factors
    k_squared = torch.einsum("ij,ij->i", kvectors, kvectors)  # Shape: (N_k,)
    gaussian_factor = torch.exp(-k_squared / (4 * kappa**2)) / k_squared  # Shape: (N_k,)
    # Compute k⋅r for all atoms at once (Shape: (N_k, N))
    k_dot_r = torch.matmul(kvectors, coords.T)  # (N_k, N)
    # Compute cos and sin terms for all atoms & k-vectors
    cos_k_dot_r = torch.cos(k_dot_r)  # (N_k, N)
    sin_k_dot_r = torch.sin(k_dot_r)  # (N_k, N)
    # Compute structure factors in a vectorized way
    sk_cos = torch.sum(q * cos_k_dot_r, dim=1)  # (N_k,)
    sk_sin = torch.sum(q * sin_k_dot_r, dim=1)  # (N_k,)
    # Compute k⋅p (Shape: (N_k, N))
    k_dot_p = torch.sum(p[:, None, :] * kvectors[None, :, :], dim=2)  # (N, N_k)
    sk_dipole_cos = torch.sum(k_dot_p.T * cos_k_dot_r, dim=1)  # (N_k,)
    sk_dipole_sin = torch.sum(k_dot_p.T * sin_k_dot_r, dim=1)  # (N_k,)
    # Compute k_a k_b (outer product) and contract with theta
    k_outer = kvectors[:, :, None] * kvectors[:, None, :]  # (N_k, 3, 3)
    h_contract_theta = torch.sum(k_outer[:, None, :, :] * t[None, :, :, :], dim=(2, 3))  # (N_k, N)
    sk_quad_cos = torch.sum(h_contract_theta * cos_k_dot_r, dim=1)  # (N_k,)
    sk_quad_sin = torch.sum(h_contract_theta * sin_k_dot_r, dim=1)  # (N_k,)
    # Compute sin and cosine terms for all unique particles at once
    #sin_term = torch.sum(sin_k_dot_r, dim=0)  # (N,)
    #cos_term = torch.sum(cos_k_dot_r, dim=0)  # (N,)
    # Compute potentials for all atoms at once
    #P_qq = torch.sum(gaussian_factor[:, None] * (cos_term * sk_cos + sin_term * sk_sin), dim=0)  # (N,)
    #P_qu = torch.sum(gaussian_factor[:, None] * (sin_term * sk_dipole_cos - cos_term * sk_dipole_sin), dim=0)  # (N,)
    #P_qt = torch.sum(gaussian_factor[:, None] * (cos_term * sk_quad_cos + sin_term * sk_quad_sin), dim=0)  # (N,)
    # Compute sin and cosine terms for all unique particles at once
    sin_term = torch.sum(sin_k_dot_r, dim=0)  # (N,)
    cos_term = torch.sum(cos_k_dot_r, dim=0)  # (N,)

    # Fix broadcasting issue
    P_qq = torch.sum(gaussian_factor[:, None] * (cos_term[None, :] * sk_cos[:, None] + sin_term[None, :] * sk_sin[:, None]), dim=0)  # ✅ Fixed
    P_qu = torch.sum(gaussian_factor[:, None] * (sin_term[None, :] * sk_dipole_cos[:, None] - cos_term[None, :] * sk_dipole_sin[:, None]), dim=0)  # ✅ Fixed
    P_qt = torch.sum(gaussian_factor[:, None] * (cos_term[None, :] * sk_quad_cos[:, None] + sin_term[None, :] * sk_quad_sin[:, None]), dim=0)  # ✅ Fixed

    # Apply final scaling factors
    P_qq *= constant
    P_qu *= constant
    P_qt *= -constant / 3

    print("CHARGE-CHARGE POTENTIAL: ", P_qq)
    print("CHARGE-DIPOLE POTENTIAL: \n", P_qu)
    print("CHARGE-QUADRUPOLE POTENTIAL: \n", P_qt)

    # Total reciprocal space potential
    P_recip = P_qq + P_qu + P_qt
    return P_recip

def get_self_interaction_potential_vectorized(coords,q,t,kappa):
    P_qq = q * (2*kappa/math.pi)
    P_qt = torch.einsum("bii->b",t) * (-4*kappa**3)/(9*math.sqrt(torch.pi))
    P_self = P_qq + P_qt
    return P_self
    
def get_kvectors(box,kcutoff):
  V = torch.abs(torch.dot(box[2],torch.linalg.cross(box[0],box[1]))) #compute volume of box
  kx = (2*math.pi)/ V * torch.linalg.cross(box[1],box[2])
  ky = (2*math.pi)/ V * torch.linalg.cross(box[2],box[0])
  kz = (2*math.pi)/ V * torch.linalg.cross(box[0],box[1])
  constant = 8*math.pi/V
  k_sq_max = kcutoff**2
  kvectors = []
  max_hkl = 5
  hkl_range = torch.arange(-max_hkl, max_hkl + 1)
  for h in hkl_range:
    for k in hkl_range:
      for l in hkl_range:
        kvec = h*kx + k*ky +l*kz
        if torch.norm(kvec) < k_sq_max and torch.norm(kvec)>0:
          kvectors.append(kvec)
  kvectors = torch.stack(kvectors)
  return kvectors

def get_electric_field(coords, q , p , t,box, kappa,kcutoff):
    kvectors = get_kvectors(box,kcutoff)
    E_real = get_real_space_field(coords,q,p,t,box,kappa)
    #print("CHARGES: ", q)
    #print("DIPOLES: ", p)
    #print("QUADRUPOLES: ", t)
    print("FIELD REAL")
    print(E_real)
    E_reciprocal = get_reciprocal_space_field(coords,q,p,t,box,kappa,kcutoff,kvectors)
    print("FIELD RECIPROCAL")
    print(E_reciprocal)
    #E_self = get_self_interaction_field(coords,q,p,t,box,kappa)
    E_self_vectorized = get_self_interaction_field_vectorized(coords,q,p,t,box,kappa)
    #print("FIELD SELF")
    #print(E_self)
    print("FIELD SELF VECTORIZED OUTPUT")
    print(E_self_vectorized)
    E = E_real + E_reciprocal - E_self_vectorized
    P_real = get_real_space_potential(coords,q,p,t,box,kappa)
    print("REAL SPACE POTENTIAL")
    print(P_real)
    #P_self = get_self_interaction_potential(coords,q,t,kappa)
    P_self_vectorized = get_self_interaction_potential_vectorized(coords,q,t,kappa)
    #print("SELF INTERACTION POTENTIAL")
    #print(P_self)
    print("SELF INTERACTION VECTORIZED OUTPUT:")
    print(P_self_vectorized)
    P_reciprocal = get_reciprocal_space_potential(coords, q, p , t ,box, kappa,kcutoff,kvectors)
    print("POTENTIAL RECIP")
    print(P_reciprocal)
    P_reciprocal_vectorized = get_reciprocal_space_potential_vectorized(coords, q, p , t ,box, kappa,kcutoff,kvectors)
    #print("POTENTIAL RECIP VECTOIZED")
    #print(P_reciprocal_vectorized)
    return E
