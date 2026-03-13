import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from scipy import optimize 

from scipy.special import lambertw
import time 
import matplotlib.colors as mcolors
import matplotlib.cm as cm

tau_d = 5.0
kappa = 0.1
alpha = 2.0

dq = 0.01
q_t = np.arange(0, 7, dq)

def RHS(q, br):
    term1 = -alpha * tau_d * (q**2) * np.exp(tau_d * (kappa*(q**4)  - (q**2)))
    term2 = (- kappa*(q**4)  + (q**2))
    return lambertw(term1, k=br) / tau_d + term2

Branches = np.arange(-3, 4, 1)
lambda_r = np.empty((len(Branches), len(q_t)))
lambda_im = np.empty((len(Branches), len(q_t)))

for i, Branch in enumerate(Branches):
    lm = RHS(q_t, Branch)
    lambda_r[i, :] = lm.real
    lambda_im[i, :] = lm.imag

def find_infind(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

if np.isinf(lambda_r[:, 1:]).any():
    idx = find_infind(lambda_r[0, :], lambda e: e == np.inf)
    start_q = q_t[idx[0]]
    x_g = lambda_r[:, idx[0]-1]
    y_g =  lambda_im[:, idx[0]-1]
else:
    start_q = np.max(q_t) 
    x_g = lambda_r[:, len(q_t)-1]
    y_g =  lambda_im[:, len(q_t)-1]

q_re = np.round(np.logspace(np.log10(start_q + dq), np.log10(100000), 20000), 4)
print(f"Need to reconstruct roots from q {start_q} to {np.max(q_re)}") if np.isinf(lambda_r).any() else None
print("Number of q values to compute: ", len(q_re))

q_t = np.arange(0, start_q, dq)


cwd = os.getcwd()
folder_path = rf'{cwd}/LambdaS_data/Roots_dq_{dq}_k{kappa}_td{tau_d}_al{alpha}'

if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
##Numerical reconstruction 

Lg = x_g + 1j * y_g

def func(q0, lm):
    q02 = q0**2
    return lm - (q02) + kappa*(q02**2) + alpha*q02*np.exp(-lm * tau_d)

def func_p(q0, lm):
    q02 = q0**2
    return 1 - alpha*q02*tau_d*np.exp(-lm * tau_d)


Nlambda_r = np.zeros((len(Branches), len(q_re) ))
Nlambda_im = np.zeros((len(Branches), len(q_re)))


for i, q in enumerate(q_re):
    roots = []
    for w in Lg:
        try:
            root = optimize.newton(lambda w_: func(q, w_), w, fprime=lambda w_: func_p(q, w_))
            roots.append(root)
        except RuntimeError:
            pass
            
    roots = np.array(np.unique(np.round(roots, decimals=5))) 
    Re_lm = np.real(roots)
    Im_lm = np.unique(np.imag(roots))
    
    Nlambda_r[:, i] = np.array(Re_lm) 
    Nlambda_im[:, i] = np.array(Im_lm)
    
    x_g, y_g = np.meshgrid(Re_lm, Im_lm)
    Lgg = x_g + 1j * y_g 
    Lg = Lgg.flatten()
    if np.mod(i, 10000) == 0: 
        print(f"Number of roots for q = {q}, is = {len(np.unique(roots))}")

npy_path = os.path.join(folder_path, "Nlambda_r.npy")
np.save(npy_path, Nlambda_r)
npy_path = os.path.join(folder_path, "Nlambda_im.npy")
np.save(npy_path, Nlambda_im)
print("done!")

