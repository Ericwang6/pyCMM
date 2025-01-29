import torch
from scipy import special
from torch.special import erfc
import math
from . import units, pbc
from .units import EPSILON0 , ELE_CHG 
from .pbc import applyPBC
import numpy as np
torch.set_printoptions(profile="full")
# This file contains the Ewald Summation for computing Long range interactions
def short_range(coords, q , p, t, box, kappa, rcutoff):
  U_cc =0
  U_cd = 0
  U_dd = 0
  U_ct = 0
  U_dt =0
  U_tt = 0
  constant = 1/(4*math.pi * EPSILON0) 
  N = len(coords) #atoms
  boxInv = torch.linalg.inv(box)  # Inverse of the simulation box matrix
  for i in range(N):
    for j in range(i+1,N):
      drVec = coords[j] - coords[i]
      r_ = applyPBC(drVec.unsqueeze(0), box, boxInv).squeeze(0)
      r = torch.norm(r_) #Get distance between two points.
      rhat = r_ / r
      if r < rcutoff:
        r2 = r**2
        r3 =r2*r
        r4 = r3*r
        r5 = r3*r2
        r7 = r5*r2
        r9 = r7*r2
        kappa2 = kappa*kappa
        kappa3 = kappa2*kappa
        delta = torch.eye(3) #kronecker delta
        ########|||||Ewald screening functions are represented by f_ |||||#######
        f_0 = erfc(r * kappa) 
        f_1 = (2*kappa*r/math.pi * torch.exp(-kappa2 * r2)) + f_0
        f_2 = 4 * kappa3/math.sqrt(math.pi) * torch.exp(-kappa2 * r2)/r2
        f_3 = r3 * f_2
        f_4 = 8 * kappa/math.pi * (kappa2 * r2 + 1)/r4 * torch.exp(-kappa2 * r2)
        f_5 = r3 * f_4 - 3 * r * f_2
        f_6 = 2*(kappa2 + 1/r2) * f_4 + 4*f_2/r4
        #######|||||Multipole tensors are represented by T_. Also implemented are the D_ tensors|||||#######
        T_ = 1/r
        T_a = -rhat/r3
        T_ab = (3 * torch.outer(r_,r_) - r2 * delta)/r5
        #################################################################
        term1_abg = 15 * torch.einsum("i,j,k->ijk", r_,r_,r_)
        term2_abg = -3 * r2 *(
                    torch.einsum("i,jk->ijk", r_, delta)+
                    torch.einsum("j,ik->ijk", r_, delta)+
                    torch.einsum("k,ij->ijk", r_, delta)
        )
        T_abg = -(term1_abg + term2_abg) / r7
        #################################################################
        term1_abgd = 105 * torch.einsum("i,j,k,l -> ijkl", r_,r_,r_,r_)
        term1 = torch.einsum("i,j,kl->ijkl", r_, r_, delta)  # R_ij,a R_ij,b k_gd
        term2 = torch.einsum("i,k,jl->ijkl", r_, r_, delta)  # R_ij,a R_ij,g k_bd
        term3 = torch.einsum("i,l,jk->ijkl", r_, r_, delta)  # R_ij,a R_ij,d k_bg
        term4 = torch.einsum("j,k,il->ijkl", r_, r_, delta)  # R_ij,b R_ij,g k_ad
        term5 = torch.einsum("j,l,ik->ijkl", r_, r_, delta)  # R_ij,b R_ij,d k_ag
        term6 = torch.einsum("k,l,ij->ijkl", r_, r_, delta)  # R_ij,g R_ij,d k_ab
        term2_abgd = -15 * r2 * (term1 + term2 + term3 + term4 + term5 + term6)
        term3_abgd = 3 * r4 * (
          torch.einsum("ij,kl->ijkl", delta, delta) +
          torch.einsum("ik,jl->ijkl", delta, delta) +
          torch.einsum("il,jk->ijkl", delta, delta)
        )
        T_abgd = (term1_abgd +term2_abgd + term3_abgd)/r9 #final term
        D_ab = torch.einsum("i,j->ij", r_,r_)
        D_abgd = 0 
        ###charge-charge###
        U_cc += (q[i] * q[j])*f_0*T_    
        ###charge-dipole###
        U_cd += -f_1*(q[i]*torch.dot(T_a,p[j]) - q[j]*torch.dot(T_a,p[i]))
        ###dipole-dipole###
        U_dd += -(f_1* torch.dot(p[i],torch.matmul(T_ab,p[j])) + f_2 *torch.dot(p[i],torch.matmul(D_ab,p[j])))
        ###charge-quadrupole###
        U_ct += f_1*(q[i]*torch.sum(T_ab * t[j]) + q[j]* torch.sum(T_ab*t[i]))  + f_2*(q[i] * torch.sum(D_ab * t[j]) + q[j] * torch.sum(D_ab * t[i]))
        ###dipole-quadrupole###
        ###quadrupole-quadrupole###

        print(f"Atom {i}-{j}: r = {r}, q[i] = {q[i]}, q[j] = {q[j]},"
              f" p[i] = {p[i]}, p[j] = {p[j]} U_cc = {U_cc},"
              f"U_cd = {U_cd} , U_dd = {U_dd}, U_ct = {U_ct},"
              f"U_dt = {U_dt}, U_tt = {U_tt}")
 # U_cc *= constant
 # U_cd *= constant
 # U_dd *= constant
  U_ct *= 1/3
  U_dt *= 1/3
  U_tt *= 1/9
  U_s = U_cc + U_cd + U_dd + U_ct
  return U_s

def long_range(coords, q, p, t, box, kappa, kcutoff):
  V = torch.abs(torch.dot(box[2],torch.linalg.cross(box[0],box[1]))) #compute volume of box
  constant =  (2 * V * EPSILON0)
  kx = (2*math.pi)/ V * torch.linalg.cross(box[1],box[2])
  ky = (2*math.pi)/ V * torch.linalg.cross(box[2],box[0])
  kz = (2*math.pi)/ V * torch.linalg.cross(box[0],box[1])
  k_sq_max = kcutoff**2
  kvectors = []
  max_hkl = 5
  hkl_range = torch.arange(-max_hkl, max_hkl + 1)
  print(f"K_SQ_MAX: {k_sq_max}, KCUTOFF: {kcutoff}, KAPPA: {kappa}, V: {V}, kx: {kx}, ky: {ky}, kz: {kz}")
  for h in hkl_range:
    for k in hkl_range:
      for l in hkl_range:
        kvec = h*kx + k*ky +l*kz
        if torch.norm(kvec) < k_sq_max and torch.norm(kvec)>0:
          kvectors.append(kvec)
  kvectors = torch.stack(kvectors)
  #Precalculating gaussian factors
  k_squared = torch.sum(kvectors ** 2, dim=1)
  gaussian_factor = torch.exp(-k_squared /(4*kappa**2)) / k_squared
  #Calculating all structure factors
  sk_cos = torch.zeros(len(kvectors))
  sk_sin = torch.zeros(len(kvectors))
  sk_dipole_cos = torch.zeros(len(kvectors))
  sk_dipole_sin = torch.zeros(len(kvectors))
  sk_quad_cos = torch.zeros(len(kvectors))
  sk_quad_sin = torch.zeros(len(kvectors))
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
  ###Charge–charge### 
  cc_ene = (4*math.pi/ V)* torch.sum(gaussian_factor * (sk_cos**2 + sk_sin**2))
  ###Charge–dipole### 
  cd_ene = (8 * math.pi / V) * torch.sum(gaussian_factor* (sk_sin * sk_dipole_cos - sk_cos * sk_dipole_sin))
  ###Dipole-dipole### 
  dd_ene = (4 * math.pi / V) *torch.sum(gaussian_factor * (sk_dipole_cos**2 + sk_dipole_sin**2))
  ###charge-quadrupole###
  ct_ene = -(8*math.pi/V) * torch.sum(gaussian_factor * (sk_cos *sk_quad_cos + sk_sin * sk_quad_sin)/3)
  ###dipole-quadrupole###
  dt_ene = (8*math.pi/V) * torch.sum(gaussian_factor*(sk_dipole_sin * sk_quad_cos - sk_dipole_cos * sk_quad_sin)/3)
  ###quadrupole-quadrupole###
  tt_ene = (4*math.pi/V) * torch.sum(gaussian_factor*(sk_quad_cos**2 + sk_quad_sin**2)/9)
  #####Final reciprocal energy#####
  U_l = cc_ene + cd_ene + dd_ene + ct_ene + dt_ene + tt_ene
  print("KVECTORS: \n",kvectors)
  print("k_squared values: ", k_squared)
  print("GAUSSIAN FACTOR: \n", gaussian_factor)
  print("CHARGE CHARGE ENERGY: ", cc_ene)
  print("CHARGE DIPOLE ENERGY: ", cd_ene)
  print("DIPOLE DIPOLE ENERGY: ", dd_ene)
  print("CHARGE QUADRUPOLE ENERGY: ", ct_ene)
  print("DIPOLE QUADRUPOLE ENERGY: ", dt_ene)
  print("QUADRUPOLE QUADRUPOLE ENERGY: ", tt_ene)
  return U_l


def self_interaction(coords, q , p, t, kappa):
#Self interaction energy. Subtracted from total Ewald energy.
  constant =  (1/(4*math.pi *EPSILON0))*(1/math.sqrt(2*math.pi)) 
  U_q, U_d, U_cq, U_t = 0, 0, 0, 0
  N = len(coords)
  for i in range(N):
    #monopole
    U_q += q[i]**2
    #dipole
    U_d += torch.sum(p[i]**2)
    #charge-quadrupole
    U_cq += q[i] * torch.trace(t[i])
    #quadrupole
    U_t += torch.sum(t[i]*t[i])
  #multiplying with correct constants
  U_q *= kappa/math.sqrt(math.pi)
  U_d *= 2 * kappa**3 /(3 * math.sqrt(math.pi))
  U_cq *= 2*kappa**3/(3 * math.pi)
  U_t *= 8*kappa**5/(45*math.pi)
  #total self-interaction energy
  U_self = U_q + U_d + U_cq + U_t
  #U_self = constant * U_self
  print("U_MONO: ", U_q)
  print("U_DIPOLE: ", U_d)
  print("U_CHARGE_QUADRUPOLE: ", U_cq)
  print("U_QUADRUPOLE: ", U_t)
  return U_self


def compute_ewald(coords, q ,p, t,box, kappa, rcutoff, kcutoff):
#Total energy = short range + long range - self interaction
  print("CHARGES: ", q)
  print("DIPOLES: ", p)
  print("QUADRUPOLES: ", t)
  U_s = short_range(coords, q,p,t, box, kappa, rcutoff) 
  U_l =  long_range(coords, q,p,t, box, kappa, kcutoff)  
  U_self = self_interaction(coords, q,p,t, kappa)
  U_ewald = U_s + U_l - U_self
  print(f"FINAL ENERGIES: U_s = {U_s} , U_l = {U_l} , U_self = {U_self}, total = {total}")
  return U_ewald



