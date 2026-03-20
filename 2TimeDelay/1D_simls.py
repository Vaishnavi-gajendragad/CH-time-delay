import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import time
from scipy import signal

N = 512
L = 50
dx = L/N

x = np.arange(0, L, dx)

def Wavenumbers1D(N, L):
    const_q = (2*np.pi/(L))
    return np.fft.fftfreq(N, d=(1/(N * const_q))), np.fft.rfftfreq(N, d=(1/(N * const_q))), int(N//3)

q, q_pos, q_p = Wavenumbers1D(N, L)

dealiase_1D = np.ones(len(q_pos))
dealiase_1D[q_p + 1:] = 0

#time-stepping parameters
dt = 0.0001
t_max = 500
it = np.arange(0, t_max+1, dt)

kappa = 1
mean = 0

#activity parameters
alpha1 = 5
alpha2 = 2

td1 = 3
td2 = 7

delay_step1 = int(td1/dt)
delay_step2 = int(td2/dt)

diff = delay_step2 - delay_step1


ss = 500000

print(f"Resolution N: {N}, length L: {int(L/np.pi)}pi, dx: {np.round(dx, 6)}")
print(f"\nq max: {np.max(q)}, q min: {np.min(abs(q))}")
print(f"\ndealiasing from q: {q[q_p +1]} to {q[-q_p] }")
print(f"\nkappa: {kappa}")
print(f"\nalpha1: {alpha1}, alpha2: {alpha2}")
print(f"\ntime step: {dt}, total steps: {len(it)}")
print(f"\ndelay step 1: {delay_step1}, tau_d: {td1}")
print(f"\ndelay step 2: {delay_step2}, tau_d: {td2}")
print(f"\nMean composition: {mean}")
print(f"Saving data for {ss}")

cwd = os.getcwd()
folder_path = f'{cwd}/N{N}_L{L}_dt{dt}/Model_1/al1({alpha1})_al2({alpha2})_td1({td1})_td2({td2})'
os.makedirs(folder_path, exist_ok=True)

def CH_ETD_solver(phi_in, alpha1, alpha2, delay_step1, delay_step2):
    phi_mat = np.empty((delay_step2 + 1, N))
    phi_mat[0, :] =  (phi_in ) - np.mean(phi_in)
    
    phi_save = np.zeros((int(ss), N))
    
    s = 0 
    js = 0 

    c = -(kappa*(q2**2))
    K1 = np.exp(c * dt)
    K2 = (K1 - 1) / c
    K2[0] = 0  

    def phi_ETD1(phi_n, alpha1, alpha2, phi_d1, phi_d2):                                                                                                                                             
        F_n = -q2 * (-np.fft.rfft(phi_n) + np.fft.rfft(phi_n**3) + alpha1*np.fft.rfft(phi_d1) + alpha2*np.fft.rfft(phi_d2))
        return (np.fft.rfft(phi_n) * K1) + (F_n * K2)
        
    for j, i in enumerate(it):
            if j < delay_step1:
                rubbish = np.random.rand(N)*0.01
                phi_mat[j + 1] = np.fft.irfft(dealiase_1D * phi_ETD1(phi_mat[j], 0, 0, rubbish, rubbish), N)
            elif delay_step1 < j < delay_step2: 
                rubbish = np.random.rand(N)*0.01 
                phi_d1 = phi_mat[j - delay_step1, :]
                phi_q = phi_ETD1(phi_mat[j], alpha1, 0, phi_d1, rubbish)
                phi_mat[j + 1] = np.fft.irfft(dealiase_1D * phi_q, N)
            elif j > delay_step2:
                #j_n = s
                j_n = (s + delay_step2) % (delay_step2 + 1) #current time step 
                s1 = (s + diff)%(delay_step2 + 1)
                phi_d2 = phi_mat[s, :]
                phi_d1 = phi_mat[s1, :]

                phi_q = phi_ETD1(phi_mat[j_n], alpha1, alpha2, phi_d1, phi_d2)
                
                s = (s + 1)%(delay_step2 + 1)              #next time step
                
                j_next = (s + delay_step2) % (delay_step2 + 1)
                #j_next = s
                phi_mat[j_next] = np.fft.irfft(dealiase_1D  * phi_q, N)   
                
                if np.isnan(phi_mat[j_next]).any():
                    print("Nan found - aborting")
                    return 
                if j % 100000 == 0:
                    print(f"Processing j: {j}, {np.round(j/len(it), 2) * 100}% completed")
                if j >= len(it) - ss:
                    phi_save[js, :] = phi_mat[j_next, :]
                    js = js + 1
    return j, phi_save

phi_in = np.random.rand(N)*0.01
it_max, phi_steady = CH_ETD_solver(phi_in, alpha1, alpha2,  delay_step1, delay_step2)
np.save(os.path.join(folder_path, 'phi_steady.npy'), phi_steady)
np.save(os.path.join(folder_path, 'phi_in.npy'), phi_in)
