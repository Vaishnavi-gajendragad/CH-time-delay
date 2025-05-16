import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fft import fft2, ifft2
from matplotlib import rc
import os
import time


A = 20
font = {'family' : 'serif', 'weight' : 'bold'}
rc('font', family='serif')

N = 512 
a = 4
L = a*np.pi
dx = L / N
x = np.arange(0, L, dx)

dt = 0.001
t_max = 50
it = np.arange(0, t_max, dt)

kdx = 2
kappa = (kdx * dx)**2
#kappa = 1
kappa = np.round(kappa, 5)

alpha = 0

tau_d = 0.01
delay_step = int(tau_d/dt)
#tau_d = delay_step * dt 

mean = 0

print(f"Resolution N: {N}, length L: {int(L/np.pi)}\\pi, dx: {np.round(dx, 6)}")
print(f"\nkappa: {kappa}, alpha: {alpha}")
print(f"\nalpha: {alpha}")
print(f"\ntime step: {dt}, total steps: {len(it)}")
print(f"\ndelay step: {delay_step}, tau_d: {tau_d}")
print(f"\nMean composition: {mean}")

const_q = (2*np.pi/(L)) 

qt_x =  const_q * np.single(np.concatenate((np.arange(0, N/2 + 1), np.arange(-(N/2 - 1), 0))))
qt_y =  const_q * np.single(np.concatenate((np.arange(0, N/2 + 1), np.arange(-(N/2 - 1), 0))))

q_x, q_y = np.meshgrid(qt_x, qt_y)
q2 = q_x**2 + q_y**2 

qx_p = const_q*len(qt_x)//3 
qy_p = const_q*len(qt_y)//3

Mat  = np.ones((N, N))

def dealiasing(kx_inf, ky_inf, k_x, k_y, N_x, N_y, M):
    for i in range(N_x):
        for j in range(N_y):
            D = (k_x[i, j] / kx_inf) ** 2 + (k_y[i, j] / ky_inf) ** 2
            if D > 1:
                M[i, j] = 0
    return M

IIKill = dealiasing(qx_p, qy_p , q_x, q_y, N, N, Mat)

cwd = os.getcwd()
if alpha == 0: 
    Sup_folder_name = rf'ETDDATA_N{N}_L{a}pi_mean{mean}_dt{dt}_k{kdx}dx'
    folder_path = rf'{cwd}/Passive/{Sup_folder_name}/Data'
    if not os.path.exists(folder_path):
        print("Folder doesn't exist, creating one.. ")
        os.makedirs(folder_path)
        
    png_folder_path = rf'{cwd}/Passive/{Sup_folder_name}/plots'
    if not os.path.exists(png_folder_path):
        print("Folder doesn't exist, creating one.. ")
        os.makedirs(png_folder_path)
else: 
    td_name = rf'td_{tau_d}'
    Sup_folder_name = rf'ETDDATA_N{N}_L{a}pi_m{mean}_dt{dt}_k{kdx}dx_td{tau_d}_al{alpha}'
    #folder_path = rf'{cwd}/Active/Test/{Sup_folder_name}/Data'
    #if not os.path.exists(folder_path):
    #    print("Folder doesn't exist, creating one.. ")
    #    os.makedirs(folder_path)
        
    png_folder_path = rf'{cwd}/Active/Test/{td_name}/{Sup_folder_name}'
    if not os.path.exists(png_folder_path):
        print("Folder doesn't exist, creating one.. ")
        os.makedirs(png_folder_path)

print(Sup_folder_name)

def CH_ETD_solver(phi_init, alpha, delay_step):
    phi_mat = np.zeros((delay_step + 1, *phi_init.shape), dtype=phi_init.dtype)
    phi_mat[0] = phi_init
    s = 0

    def phi_ETD1(phi_n, alpha, phi_d):
        c = -(kappa * q2**2)
        K1 = np.exp(c * dt)
        K2 = (K1 - 1) / c
        K2[0] = 0
        F_n = -q2 * ( -np.fft.fft2(phi_n) + np.fft.fft2(phi_n**3) + alpha*np.fft.fft2(phi_d))
        return (np.fft.fft2(phi_n) * K1) + (F_n * K2)

    def save_fig(data, j, title, path):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.set_title(title)
        im = ax.imshow(data, cmap='Spectral', aspect='equal')
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cbar.ax.tick_params(labelsize=12)
        ax.axis('off')
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    phi = phi_init.copy()
    #np.save(os.path.join(folder_path, 'phi_t0.npy'), phi)
    save_fig(phi, 0, r"t = 0", os.path.join(png_folder_path, 'phi_t0.png'))

    for j, i in enumerate(it):
        if j < delay_step:
            rubbish = np.random.rand(N, N)*0.1
            phi_mat[j + 1] = np.fft.irfft2(IIKill * phi_ETD1(phi_mat[j], 0, rubbish), s=np.shape(phi))
        else:
            phi_d = phi_mat[s]
            #j_n = s
            j_n = (s + delay_step) % (delay_step + 1) #current time step 
            phi_q = phi_ETD1(phi_mat[j_n], alpha, phi_d)
            
            s = (s + 1)%(delay_step + 1)              #next time step
            j_next = (s + delay_step) % (delay_step + 1)
            #j_next = s
            phi_mat[j_next] = np.fft.irfft2(IIKill * phi_q, s=np.shape(phi))    
            if np.isnan(phi_mat[j_next]).any():
        	     break 
        if j % 1000 == 0:
            print("j:", j)
            np.save(os.path.join(folder_path, f'phi_t{j}.npy'), phi_mat[s])
        if j % 1000 == 0:
            save_fig(phi_mat[s], j, rf"t = {round(j * dt, 4)}", os.path.join(png_folder_path, f'phi_t{j}.png'))

phi = np.random.rand(N, N)*0.1 + mean

CH_ETD_solver(phi, alpha, delay_step) #run it 
