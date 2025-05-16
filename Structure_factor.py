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
kdx = 1.0

dt = 0.00001
t_max = 10

it = np.arange(0, t_max, dt)

alpha = 2.0
mean = 0
tau_d = 0.01
const_q = (2*np.pi/(L)) 

qt_x =  const_q * np.single(np.concatenate((np.arange(0, N/2 + 1), np.arange(-(N/2 - 1), 0))))
qt_y =  const_q * np.single(np.concatenate((np.arange(0, N/2 + 1), np.arange(-(N/2 - 1), 0))))

q_x, q_y = np.meshgrid(qt_x, qt_y)

qs_grid = np.sqrt(q_x**2 + q_y**2)
q2 = q_x**2 + q_y**2 

q_id = np.round(qs_grid).astype(int)
#print(np.diag(qs_grid))
#bins = np.unique(q_id)
q_pos = np.sqrt( (qt_x[qt_x > 0])**2 + (qt_y[qt_y > 0])**2)
#print(np.round(q_pos))
S_k = np.zeros((len(it), N))

cwd = os.getcwd()


if alpha == 0: 
    Sup_folder_name =  rf'{cwd}/ETDDATA_N{N}_L{a}pi_mean{mean}_dt{dt}_k{kdx}dx'
    #folder_path = rf'{Sup_folder_name}/S_k'
    png_folder_path = rf'{Sup_folder_name}/S_k'
    if not os.path.exists(png_folder_path):
        print("Folder doesn't exist, creating one.. ")
        os.makedirs(png_folder_path)
else: 
    #/data.lmp/vgajendragad/CH_td/Active/ETDDATA_N512_L4pi_mean0_dt1e-05_k1.0dx_td0.01_al2.0
    td_name = rf'td{tau_d}'
    data_lmp_filepath = rf"/data.lmp/vgajendragad/CH_td/Active"
    Data_dir = rf"ETDDATA_N{N}_L{a}pi_mean{mean}_dt{dt}_k{kdx}dx_td{tau_d}_al{alpha}"
    Sup_folder_name = rf'{data_lmp_filepath}/{Data_dir}/Data'    
    if not os.path.exists(Sup_folder_name):
        print("Data folder doesn't exist")
    png_folder_path = rf'{cwd}/Active/Cluster_data/{td_name}/{Data_dir}/S_k'
    if not os.path.exists(png_folder_path):
        print("Folder doesn't exist, creating one.. ")
        os.makedirs(png_folder_path)


def save_fig(data, j, title, path):
        fig, ax = plt.subplots(dpi=300)
        ax.set_title(title)
        im = ax.loglog(q_pos[0:N//3], data, color='r', lw='2')
        ax.set_xlabel("q", fontsize='18')
        ax.set_ylabel(r"$S_k$", fontsize='18')
        #cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        #cbar.ax.tick_params(labelsize=12)
        #ax.axis('off')
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
for s, t in enumerate(it[::10000]):
    print(s, np.round(t, 5))
    filename = os.path.join(rf'{Sup_folder_name}', f'phi_t{s * 10000}.npy')
    phi_k = np.fft.fft2(np.load(filename))
    phi_k2 = np.abs(phi_k)**2
    for k in range(N):
        S_k[s, k] = phi_k2[q_id == k].sum()
    if s % 10000:
        save_fig(S_k[s, 0:N//3], s, rf"t = {np.round(t, 5)}", os.path.join(png_folder_path, f'Sk_t{s}.png'))
