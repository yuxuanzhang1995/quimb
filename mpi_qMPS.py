#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:45:05 2021

@author: zhangslaptop
"""
from quimb import *
from quimb.tensor import *
from random import*
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import jaxlib
#pip install --no-deps -U -e .
#import cotengra as ctg
import jax.numpy as jnp
#import tensorflow
import scipy
from mpi4py import MPI


def pre_meas():
    '''
    A circuit to prepare unitary circuit that we use for density operator
    '''
    rng = np.random.default_rng()
    params = rng.uniform(low=1.5,high=2,size = 1)
    circ = qtn.Circuit(2)
    if T == 0:
        m = np.exp(-1/1e-32)
    else:
        m = np.exp(-1/T)
    circ.apply_gate('RX', *params*m, 0,
            gate_round=None,parametrize=True)
    circ.apply_gate('CNOT',0,1,
            gate_round=None,parametrize=False)
    return circ


def density_matrix(unitary_circ):
    '''
    Here we prepare the density matrix
    '''
    a = unitary_circ.uni
    ini = np.zeros([4,4])
    ini[0][0]=1
    ini = ini.reshape(2,2,2,2)
    ind = 'k0','bc0','k1','bc1'
    den = qtn.Tensor(ini,ind,tags='den')
    b = a.conj().reindex({'k0':'bc0','b0':'kc1','b1':'kc0','k1':'bc1'})
    TN = a & den & b
    TN = TN.reindex({'b0':'x','kc1':'x'})
    #ind2 = 'b1',
    #u = qtn.Tensor(np.array([1,1]),ind2)
    #TN1 = u & TN
    return TN
para = np.array([4.81542392, 5.75452388, 2.49307931, 3.56823768, 2.88486158,
       3.26971195, 1.6737704 , 0.64848863, 1.04215001, 0.42668905,
       6.15501872, 4.45570347, 4.71239167, 4.71230547, 4.71247841])
rng = np.random.default_rng()
#x = rng.uniform(high=2*np.pi,size = 15)
def qmps_f(L=16, in_depth=2, n_Qbit=3, data_type='float64', qmps_structure="pollmann", canon="left", seed_init=10):

    seed_val=seed_init

    list_u3=[]
    n_apply=0
    circ = qtn.Circuit(L)

    if canon=="left":
        for i in range(0,L-n_Qbit,1):#L-n_Qbit is 
            Qubit_ara=i+n_Qbit
            if qmps_structure=="pollmann":
                n_apply, list_u3=range_su4_pollmann(circ, i, n_apply, list_u3, in_depth, n_Qbit,data_type,seed_val, Qubit_ara)

   #psi=convert_wave_function_to_real(psi, L, list_u3)
    return circ, list_u3

def range_su4_pollmann(circ, i_start, n_apply, list_u3, depth, n_Qbit,data_type,seed_val,Qubit_ara):
    gate_round=None
    if n_Qbit==0: depth=1
    if n_Qbit==1: depth=1
    c_val=0
    for r in range(depth):
        for i in range(i_start, i_start+n_Qbit, 1):
         #print("U_e", i, i + 1, n_apply)
            param = para+rng.uniform(high=0.2,size = 15)
            #param = qu.randn(15,scale=0.0, dist='uniform')
            circ.apply_gate('SU4',*param,i,i+1,gate_round=f'{n_apply%(n_bond*depth)}', parametrize=True,contract=False)#, gate_opts={tags:'SU4',f'G{n_apply}',f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
            list_u3.append(f'G{n_apply}')
            n_apply+=1
            c_val+=1

    return n_apply, list_u3
def pollmann_mpdo(cir,n_phys,n_bond,burnin=0):
    
    n_total = n_bond + n_phys

    dm =  density_matrix(pre_meas())
    state_vcl = np.array([1.,0.])
    mps = cir.uni
    mps_c = mps.H
    
    mps = mps.reindex({ f"k{i}":f"b_in{i}"  for i in range(n_bond)})
    mps = mps.reindex({ f"b{n_total-i-1}":f"b_out{i}"  for i in range(n_bond)})
    mps = mps.reindex({ f"k{i+n_bond}":f"p_in{i}"  for i in range(n_phys)})
    mps = mps.reindex({ f"b{i}":f"p_out{i}"  for i in range(n_phys)})

    mps_c = mps_c.reindex({ f"k{i}":f"bc_in{i}"  for i in range(n_bond)})
    mps_c = mps_c.reindex({ f"b{n_total-i-1}":f"b_out{i}"  for i in range(n_bond)})
    mps_c = mps_c.reindex({ f"k{i+n_bond}":f"pc_in{i}"  for i in range(n_phys)})
    if burnin!=0:
        mps_c = mps_c.reindex({ f"b{i}":f"p_out{i}"  for i in range(burnin)})
    mps_c = mps_c.reindex({ f"b{i}":f"pc_out{i}"  for i in range(burnin,n_phys)})
    
    tn = mps & mps_c
    for i in range (n_phys):
        dm0 = dm.reindex({'b1':f'p_in{i}','kc0':f'pc_in{i}'})
        tn = tn & dm0
    for i in range (n_bond):
        
        inds = f'b_in{i}',
        s_left = qtn.Tensor(state_vcl,inds,tags='sl')
        
        # for bra:
        inds_c = f'bc_in{i}',
        sc_left = qtn.Tensor(state_vcl.conj(),inds_c,tags='scl')
        
        tn = tn&s_left&sc_left
    return tn

def prob(V):
    p1 = V['RX','GATE_0'][0].data[0][0]**2
    p2 = 1-p1
    prob_list = [p1,p2]
    return prob_list
# free energy

def entropy(prob_list):
    """
    Returns the entropy
    """ 
    # avoiding NaN in numpy.log() function
    new_prob_list = []
    for j in prob_list:
        new_prob_list.append(j+1.e-10)
    s_list = [-p*jnp.log(p) for p in new_prob_list]
    s = sum(s_list) # entropy
    return s.real

    
def steady_state(V): # input: a tensor network
    tensor0 = V.tensors[:n_bond*Depth]
    tensor2 = v_opt.tensors[2*n_bond*Depth*n_phys:2*n_bond*Depth*n_phys+7]
    tn0 = tensor0[0]
    den = tensor2[0]
    for i in range (1,len(tensor0)):
        tn0 = tn0&tensor0[i]
    u = np.reshape(tn0.contract(all).data,(2,4,2,4))
    dx = np.asarray(prob(V))
    W1 = np.einsum('abcd,c,eick->abdeik',u,dx,np.conj(u))
    W2 = np.einsum('abdeik->abiedk',W1)
    transfer_mat = np.einsum('abad->bd',np.reshape(W2,[2,4**n_bond,2,4**n_bond]))
    eig_vals,eig_vecs = scipy.linalg.eig(transfer_mat)
    idx = np.where(np.abs(1-abs(eig_vals))<1e-5)[0][0] # index of steady-state
    steady_den = np.reshape(eig_vecs[:,idx],[2**n_bond,2**n_bond]) # steady-state density matrix
    bvecl = steady_den/np.trace(steady_den) # normalization of steady-state
    return bvecl #output: normalized tensor

def loss(V,H,T):
    """
    Represents the free energy function
    """
    TN = (V & H).rank_simplify()
    prob_list = prob(V)
    E = np.real((TN).contract(all, optimize='auto-hq',backend = 'jax'))
    S = entropy(prob_list)
    F = E- T*S
    return F

# Hamiltonian functions
def tfim_ising(J, g, h, N, bc):
    """
    Unit-cell matrix product operator of Transverse 
    Field Ising model (TFIM). 
    """
    if bc == 'infinite':
        J, g, h = J/(N-1),g/N,h/N
    elif bc == 'finite':
        J, g, h = J/N,g/N,h/N
    # Pauli matrices 
    sigmax = np.array([[0.*1j, 1], [1, 0.]])
    sigmay = np.array([[0., -1j], [1j, 0.]])
    sigmaz = np.array([[1, 0.], [0., -1]])
    id = np.eye(2)
    
    # structure of TFIM Ising model MPO unit cell
    H = np.zeros((3, 3, 2, 2))+0j
    H[0, 0] = H[2, 2] = id
    H[1, 0] = sigmaz
    H[2, 0] = (g * sigmaz + h * sigmax)
    H[2, 1] = (J * sigmaz)
    H1 = np.einsum('abcd->cbda',H)
    return H1 

def H(Hamiltonian,n,j):
    Ham = np.zeros((2, 3, 2, 3))
    inds4 = f'p_out{j}',f'H{n+1}',f'pc_out{j}',f'H{n}'
    #if n>2:#burinin

    H = qtn.Tensor(Hamiltonian,inds4,tags='H')
    #else:
        #H = qtn.Tensor(Ham,inds4,tags='H')
    return H

def H_contract(Hamiltonian,N,H_bvecl,H_bvecr):
    """
    Returns tensor contractions of Hamiltonian
    """
    TN_list = [H(Hamiltonian,n,j) for n,j in zip(range(N),range(N))]
    
    # for Hamiltonian
    inds4 = 'H0',
    H_left = qtn.Tensor(H_bvecl,inds4,tags='Hl')
    inds5 = f'H{N}',
    H_right = qtn.Tensor(H_bvecr,inds5,tags='Hr')

    # tenor contractions
    TN0 = TN_list[0]
    for j in range(1,len(TN_list)):
        TN0 = TN_list[j] & TN0
    
    TN = H_left & H_right & TN0
    return TN

def Smart_infgate(qmps,T,Depth):
  dic_mps=load_from_disk(f"Store/infgateInfo_T{np.round(T,2)}_B{n_bond}_L{Depth}_J{J}V{V}h{h}g{g}")
  for i in dic_mps:
      #print (i, dic_mps[i])
      t=qmps[i]
      t = t if isinstance(t, tuple) else [t]
      for j in range(len(t)):
       try:  
             if len(t)==1:
              qmps[i].params=dic_mps[i]
             else: 
              qmps[i][j].params=dic_mps[i]

       except (KeyError, AttributeError):
                   pass  # do nothing!
  return qmps



def  save_info_QMPS(qmps,T,Depth):

 tag_list=list(qmps.tags)
 tag_final=[]
 for i_index in tag_list: 
     if i_index.startswith('R'): tag_final.append(i_index)


 dic_mps={}
 for   i   in   tag_final:
   t = qmps[i]
   t = t if isinstance(t, tuple) else [t]
   dic_mps[i] = t[0].params
 save_to_disk(dic_mps, f"Store/infgateInfo_T{np.round(T,2)}_B{n_bond}_L{Depth}_J{J}V{V}h{h}g{g}")
    
# Hamiltonian functions
from hamiltonian import model_mpo
J = 1
V = 0.5
h = 1
T = 1
global Depth

n_bond = 2
burnin = 4
n_phys = 6+burnin
g=0

#Hamiltonian = tfim_ising(J,g,h,n_phys,bc='infinite')

#MPO_origin=qtn.MPO_ham_ising(n_phys-burnin, j=4*J, bx=2*h, S=0.5, cyclic=True)
#Ham = MPO_origin.reindex({f'k{i}':f'p_out{i+burnin}' for i in range(n_phys-burnin)})
#Ham = Ham.reaindex({f'b{i}':f'pc_out{i+burnin}' for i in range(n_phys-burnin)})

Hamiltonian = model_mpo.sd_ising(J,V,h,n_phys-burnin)
#Hamiltonian = model_mpo.tfim_ising(J,g,h,n_phys-burnin)
# boundary conditions
H_bvecl = np.zeros(5)
H_bvecr = np.zeros(5)
H_bvecr[0] = 1.
H_bvecl[-1] = 1.
bdry_vecs2 = [H_bvecl,H_bvecr]

Ham = H_contract(Hamiltonian,n_phys-burnin,H_bvecl,H_bvecr)
Ham = Ham.reindex({f'p_out{i}':f'p_out{i+burnin}' for i in range(n_phys-burnin)})
Ham = Ham.reindex({f'pc_out{i}':f'pc_out{i+burnin}' for i in range(n_phys-burnin)})
L=n_bond+n_phys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
free = []
step_len = 1
ini_step = 1
for i in range (ini_step,10,step_len):
    Depth = i
    circ, tag=qmps_f(L=L, in_depth=Depth, n_Qbit=n_bond, data_type='float64', qmps_structure='pollmann', 
    canon="left",  seed_init=(randint(1,100)))
    v=pollmann_mpdo(circ,n_phys,n_bond,burnin)
    if Depth > 1:
        v = Smart_infgate(v,T,Depth-step_len)
    opt_tags = [f'ROUND_{i}' for i in range(Depth*n_bond)]
    opt_tags.append('RX')
    tnopt = TNOptimizer(
    v,                        # the tensor network we want to optimize
    loss,                     # the function we want to minimize
    loss_constants={'H': Ham,'T':T},  # supply U to the loss function as a constant TN
    autodiff_backend='jax',   # use 'autograd' for non-compiled optimization
    optimizer='tnc',     # the optimization algorithm
    shared_tags=opt_tags,
    )    
    v_opt = tnopt.optimize(2000,jac=True)
    x_result = loss(v_opt,Ham,T)
    total_x =comm.gather(x_result, root = 0)
    total_x = comm.bcast(total_x, root=0)
    idn = total_x.index(min(total_x))
    if rank == idn:
        save_info_QMPS(v_opt,T,Depth)
    free.append(total_x)
if rank == 0:
    with open('Store/free_eng_T{np.round(T,2)}_B{n_bond}_L{Depth}_J{J}V{V}h{h}g{g}.npy', 'wb') as f:
        np.save(f,free)
    
