import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fft2, ifft2
from matplotlib import rc
from scipy import optimize 
import os
import time 
from itertools import product

alpha_t = np.arange(0, 100, 0.01)
alpha_t = np.round(alpha_t, 4)

print(f"\nNumber of alpha values: {len(alpha_t)}")

start_time = time.time()
for i, alpha in enumerate(alpha_t):
    roots = []
    s = alpha * (q0**2)
    omega_guesses = np.arange(-s, s, 0.001)  
    for w in omega_guesses:
        try:
            root = optimize.newton(lambda w_: func(alpha, w_), w, fprime=lambda w_: func_p(alpha, w_))
            roots.append(root)
        except RuntimeError:
            pass
            
    filename=f"Roots_al{alpha}"
    npy_path = os.path.join(folder_path, filename)
    np.save(npy_path, np.array(roots))
    if np.mod(i, 100) == 0: 
        print(f"Number of roots for alpha = {alpha}, is = {len(np.unique(roots))}")
        
end_time = time.time()
time_taken = end_time - start_time 
print(f"Time taken to complete the loop: {time_taken} seconds")

## Plotting 

#fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
alpha_t = np.arange(0, 100, 0.01)
alpha_t = np.round(alpha_t, 4)

q0 = 0.7
tau_d = 5
kappa = 1
cwd = os.getcwd()
l = '25'

#
#alpha_t = np.round(alpha_t, 4)

Sup_folder_name = rf'Omegap_q_{q0}'
folder_path = rf'{cwd}/{Sup_folder_name}/roots'

xs_all, ys_all = [], []
xs_pos, ys_pos = [], []
xs_amp, ys_amp = [], []

for index, alpha in enumerate(alpha_t):
    filename = f"{folder_path}/Roots_al{alpha}.npy"
    roots = np.load(filename)
    if np.mod(alpha, 10) == 0:
        print("Processing alpha: ", alpha)
        
    reData = np.unique(np.round(np.real(roots), decimals=5))
    OM = reData[reData >= 0] / tau_d
    al = alpha / tau_d
    Amp = np.round((1/3) * (1 - (kappa * q0**2) - al * np.cos(OM * tau_d)), 4)
    Pos = np.where(Amp > 0)[0]

    xs_all.extend([al] * len(OM))
    ys_all.extend(OM)

    xs_pos.extend([al] * len(Pos))
    ys_pos.extend(OM[Pos])

    xs_amp.extend([al] * len(Pos))
    ys_amp.extend(Amp[Pos])
    

# plt.scatter(xs_all, ys_all, s=2, c='k')
# plt.scatter(xs_pos, ys_pos, s=2, c='yellowgreen')
# plt.xlim(0, 5)
# plt.show()
