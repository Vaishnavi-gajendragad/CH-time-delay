from multiprocessing import Pool
import numpy as np
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
import os

def process_q_wrapper(args):
    q, Lg, func, func_p = args
    roots = []
    Lambda_g = Lg.copy()
    for w in Lambda_g:
        try:
            root = optimize.newton(lambda w_: func(q, w_), w, fprime=lambda w_: func_p(q, w_))
            roots.append(root)
        except RuntimeError:
            pass

    roots = np.array(roots)
    x_g, y_g = np.meshgrid(np.unique(roots.real), np.unique(np.round(roots.imag, 10)))
    Lgg = x_g + 1j * y_g
    
    return q, Lgg.flatten()

tau_d = float(5)
kappa = 0.1
alpha = float(2)

cwd = os.getcwd()
Sup_folder_name = rf'Lambda_roots_al{alpha}_td{tau_d}_k{kappa}'


Save_folder_name = rf'LambdaS_data/Recon_roots'
folder_path = rf'{cwd}/{Save_folder_name}/Roots_k{kappa}_td{tau_d}_al{alpha}'

if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
def func(q0, lm):
    q02 = q0**2
    return lm - (q02) + kappa*(q02**2) + alpha*q02*np.exp(-lm * tau_d)

def func_p(q0, lm):
    q02 = q0**2
    return 1 - alpha*q02*tau_d*np.exp(-lm * tau_d) 


if __name__ == '__main__':
    
    # === Initial guess grid ===
    filename = f"{cwd}/LambdaS_data/{Sup_folder_name}/Roots_al{alpha}_td{tau_d}_k{kappa}_q{2.000}.npy"
    unique_roots = (np.load(filename))
    LR = np.unique(unique_roots.real)
    LI = np.unique(np.round(unique_roots.imag, 10))
    
    x_g, y_g = np.meshgrid(LR, LI)
    Lgg = x_g + 1j * y_g 
    Lg = Lgg.flatten()
    print("Number of initial guesses: ", len(Lg))

    # === Prepare q values ===
    dq = 0.001
    q_t = np.arange(2 + dq, 5 + dq, dq)
    q_t = np.round(q_t, 4)
    
    # === Multiprocessing ===
    with Pool() as pool:
        args_list = [(q, Lg, func, func_p) for q in q_t]
        for i, result in enumerate(pool.imap(process_q_wrapper, args_list)):
            q, roots = result
            roots_unique = np.unique(roots)
            filename = f"Roots_q{q:.4f}.npy"
            npy_path = os.path.join(folder_path, filename)
            np.save(npy_path, roots_unique)

            if i % 100 == 0:
                print(f"[{i}] Saved roots for q = {q:.4f} ({len(roots_unique)} roots)")

    print("All done.")


fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(14, 8), dpi=250)  # Two subplots side by side
q_t = np.arange(2.001, 5, 0.001)
q_t = np.round(q_t, 4)

l='18'

for index, q in enumerate(q_t):
    q_str = f"{q:.4f}" 
    if np.mod(index, 100) == 0:
        print(index)
    filename = f"{folder_path}/Roots_q{q_str}.npy"
    unique_roots = np.unique(np.load(filename))
    Re_lm = ((np.real(unique_roots)))
    Im_lm = (np.round(np.imag(unique_roots), 10))
    positive_indices = Re_lm > 0
    positive_Re_lm = Re_lm[positive_indices]
    positive_Im_lm = Im_lm[positive_indices]
    
    ax1.plot(q * np.ones_like(positive_Re_lm), positive_Re_lm, '.', markersize=3, color='k')
    ax2.plot(q * np.ones_like(positive_Im_lm), positive_Im_lm, '.', markersize=3, color='r')


#plt.grid()
plt.suptitle(rf"$\alpha$ = {alpha},$\tau_d$ = {tau_d}, $\kappa$ = {kappa}", fontsize='25', y=0.94)
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)

#ax1.set_xscale("log")
ax1.set_xlabel(r"$q$", fontsize=l)
ax1.set_ylabel(r"Re($\lambda$)", fontsize=l)
#ax1.tick_params(labelsize=l)

ax2.set_xlabel(r"$q$", fontsize=l)
ax2.set_ylabel(r"Im($\lambda$)", fontsize=l)
#ax2.tick_params(labelsize=l)
ax1.set_xlim(min(q_t), max(q_t))
ax2.set_xlim(min(q_t), max(q_t))
#ax2.set_ylim(-10, 10)
plt.show()
