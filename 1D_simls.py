import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from os.path import dirname, basename, join, abspath, isdir
import matplotlib
from collections import deque

matplotlib.use('Agg')


A = 20
font = {'family' : 'serif'}
rc('font', family='serif')


# ------- Input parameters ----- #
N = float(sys.argv[1])
L = float(sys.argv[2])
alpha = float(sys.argv[3])
alpha = np.round(alpha, 4)
tau_d = float(sys.argv[4])
tau_d = np.round(tau_d, 4)
k = int(sys.argv[5])
# ---------- parameters

dx = L/N

x = np.arange(0, L, dx)

const_q = (2*np.pi/(L)) 
q = np.fft.fftfreq(N, d=(1/(N * const_q)))
q2 = q**2

q_pos = np.fft.rfftfreq(N, d=(1/(N * const_q)))

q_p = int(len(q))//3

dealiase_1D = np.ones(N)
dealiase_1D[q_p + 1: -q_p] = 0

dt = 0.0001
t_max = 1000
it = np.arange(0, t_max+1, dt)

kappa = 1

delay_step = int(tau_d/dt)

mean = 0

ss = 500000

print(f"Resolution N: {N}, length L: {int(L/np.pi)}\\pi, dx: {np.round(dx, 6)}")
print(f"\nq max: {np.max(q)}, q min: {np.min(abs(q))}")
print(f"\nkappa: {kappa}, alpha: {alpha}")
print(f"\nalpha: {alpha}")
print(f"\ntime step: {dt}, total steps: {len(it)}")
print(f"\ndelay step: {delay_step}, tau_d: {tau_d}")
print(f"\nMean composition: {mean}")

cwd = os.getcwd()
folder_path = f'{cwd}/N{N}_L{L}_al{alpha}_td{tau_d}'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
    
def CH_ETD_solver(k, phi_in, alpha, delay_step):
    peak = np.zeros(10000)
    phi_mat = np.empty((delay_step + 1, N))
    phi_mat[0, :] =  (phi_in - np.mean(phi_in)) 
    phi_save = deque([], maxlen=ss)
    
    js = 0 
    s = 0
    idx = 0 
    def phi_ETD1(phi_n, alpha, phi_d):
        c = -(kappa * q2**2)
        K1 = np.exp(c * dt)
        K2 = (K1 - 1) / c
        K2[0] = 0
        F_n = -q2 * ( -np.fft.fft(phi_n) + np.fft.fft(phi_n**3) + alpha*np.fft.fft(phi_d))
        return (np.fft.fft(phi_n) * K1) + (F_n * K2)
        
    for j, i in enumerate(it):
            if j < delay_step:
                rubbish = np.random.rand(N)*0.01
                phi_mat[j + 1] = np.fft.irfft(dealiase_1D * phi_ETD1(phi_mat[j], 0, rubbish), N)
            else:
                phi_d = phi_mat[s]
                #j_n = s
                j_n = (s + delay_step) % (delay_step + 1) #current time step 
                phi_q = phi_ETD1(phi_mat[j_n], alpha, phi_d)
                
                s = (s + 1)%(delay_step + 1)              #next time step
                j_next = (s + delay_step) % (delay_step + 1)
                #j_next = s
                phi_mat[j_next] = np.fft.irfft(dealiase_1D  * phi_q, N)   
                if np.isnan(phi_mat[j_next]).any():
                    print("Nan found - aborting")
                    return 
                if j >= 100000:
		     phi_save.append(phi_mat[j_n, :].copy)
		     if j % 100000:
		         phi_stack = np.array(phi_save)
		         phi_tavg = phi_stack.mean(axis=0)
		         S_k = np.abs(np.fft.fft(phi_tavg))**2 / (N**2)
		         idq = np.argmax(S_k[:len(q_pos)])
                         q_peak[idx] = q_pos[idq] *(1/const_q)
                         print(f"processing j {j}, q peak = {q_peak[idx]}")
                         if idx >= 10 and all(q_peak[idx - k] == q_peak[idx] for k in range(1, 3)):
			     print(f"Steady state reached at {j}")
                             phi_final = np.array(phi_save)
                             break
                        idx = idx + 1
         	
    
    return j, phi_final
    
def Spectra(k, phi_SS, q_pos, dt):
        #Spec = np.empty(3)
        PHI = phi_SS
        N = PHI.shape[1]
        Nt = PHI.shape[0]
        
        # -- Structure factor -- #
        phi_tavg = PHI.mean(axis=0)
        S_k = np.abs(np.fft.fft(phi_tavg))**2/(N**2)
        idq = np.argmax(S_k[0:len(q_pos)])
        
        # --Power spectrum-- # 
        T = (Nt) * dt 
        window = signal.windows.hann(Nt)
        Omega = (2 * np.pi) * np.fft.fftfreq(Nt, d=dt)
        Om_pos = Omega[0:Nt//2]
            
        phi_kavg = np.fft.fft(PHI, axis=1).mean(axis=1)
        #phi_kavg = phi_kavg - np.mean(phi_kavg)
        power_spec = np.abs(np.fft.fft(phi_kavg*window))**2/(Nt**2)
        idw = np.argmax(power_spec[0:len(Om_pos)])

        #fig, ax = plt.subplots(1, 2, figsize=(14, 8), dpi=300)
        #fig.tight_layout(pad=3.0)
        #ax1 = ax[0]
        #ax2 = ax[1]
        
        #ax1.loglog(q_pos[0:N//3], S_k[0:N//3], color='red', lw=2, label=r'$S_q$')
        #ax1.axvline(q_pos[idq], ls='--', color='grey', lw=1) 
        #ax1.set_xlabel(r"$q$", fontsize=10)
        #ax1.legend(fontsize='12')
        #ax2.loglog(Om_pos, power_spec[0:Nt//2], color='blue', lw=2, label=r'$|\phi(q, \omega)|^2$')
        #print(np.max(power_spec[0:Nt//2]))
        #ax2.set_ylim(min(power_spec[1:]), max(power_spec[1:]))
        #ax2.axvline(Om_pos[idw], ls='--', color='grey', lw=1) 
        #ax2.set_xlabel(r"$\omega$", fontsize=10)
        #ax2.set_ylabel(r"$$", rotation=0, fontsize=10, labelpad=15)
        #ax2.legend(fontsize='12')
        #ax1.tick_params(labelsize='10')
        #ax2.tick_params(labelsize='10')
       	#full_Sk[k, :] = S_k
       	#full_wk[k, :] = power_spec 
       	#path = os.path.join(folder_path, "Spectra.png")
       	#plt.savefig(path)  
        
        #Spec[0] = k 
        #Spec[1] = q_pos[idq] 
        #Spec[2] = Omega[idw]
        #with open(os.path.join(cwd,  f"Spectra_al{alpha}_td{tau_d}.txt"), "a") as f:
    	#    f.write(f"{Spec[0]}\t{Spec[1]}\t{Spec[2]}\n")
        
        np.save(os.path.join(folder_path, f'Sq_{k}.npy'), S_k)
        np.save(os.path.join(folder_path, f'wq_{k}.npy'), power_spec)

def Plot_phi(phi_in, phi_disp, it_max, dt, ddt, fs, ts):
        t1 = it_max - 500000
        t2 = it_max
        td_nd = np.round(tau_d/kappa, 3)
        fig, ax = plt.subplots(2, 1, dpi=300)
        fig.tight_layout(pad=2.0)
        ax1 = ax[0]
        ax2 = ax[1]  
        
        plt.suptitle(rf"$\tilde\tau_d = {td_nd}$, $\alpha$ = {alpha} ", fontsize =fs)
    
        vmin = np.min(phi_disp)
        vmax = np.max(phi_disp)
        im = ax1.imshow(phi_disp[::ddt, :], cmap='Blues', aspect='auto', vmin=vmin, vmax=vmax)
    
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='2%', pad=0.2)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=fs)
        num_rows = phi_disp[::ddt].shape[0]
        num_col = phi_disp[::ddt].shape[1]
        
        yticks_values = np.linspace(0, num_rows - 1, 3)
        ytick_labels = [f"{(t1 + i * ddt) * dt:.1f}" for i in yticks_values]
        
        ax1.set_yticks(yticks_values)
        ax1.set_yticklabels(ytick_labels)
    
        xticks_values = np.linspace(0, num_col - 1, 3, dtype=int)
        xtick_labels = [f"${i*dx/np.pi:.0f}\pi$" if i*dx/np.pi != 0 else "0" for i in xticks_values]
        
        ax1.set_xticks(xticks_values)
        ax1.set_xticklabels(xtick_labels)
        ax1.tick_params(labelsize=ts)
        
        phi1 = phi_in - np.mean(phi_in)  
        
        ax2.plot(x, phi1, color='green', lw='2', label='t = 0')
        ax2.plot(x, phi_disp[phi_disp.shape[0]-2, :], color='blue', lw='2', label=f't = {it_max*dt}')
        ax2.set_xlabel("x", fontsize=fs)
        ax2.set_ylabel(r"$\phi$(x, t)", fontsize=fs, rotation='0', labelpad=10)
        ax2.tick_params(labelsize=ts)
        ax2.legend()
        filename = f"phi_ss.png"
        path = os.path.join(folder_path, filename)
        plt.savefig(path, bbox_inches='tight')
        
    

num = np.random.randint(k*20)
seeds = np.linspace(num, num*10000, 10000, dtype=int)
np.random.seed(seeds[k])
phi_in = np.random.rand(N)*0.1
#np.save(os.path.join(folder_path, f'phi_in{k}.npy'), phi_in)
it_max, phi_steady = CH_ETD_solver(k, phi_in, alpha, delay_step)
np.save(os.path.join(folder_path, f'phi_SS{k}.npy'), phi_steady)
Spectra(k, phi_steady, q_pos, dt)
#Plot_phi(phi_in, phi_steady, it_max, dt, 10, 10, 10)	

