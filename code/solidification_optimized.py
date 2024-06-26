import numpy as np
import matplotlib.pyplot as plt


#functions and repective discretized derivatives
def f_bulk(x):
    return x**2 * (1 - x)**2

def df_bulkdx(x):
    return 2*x - 6*x**2 + 4*x**3

def div_gradient(x, dx):
    tmp = np.copy(x)
    tmp[1:-1] = (x[:-2] + x[2:] - 2 * x[1:-1]) / dx**2
    return tmp

# parameters
f_0_list = [1.0, 0.1, 0.01] 
K_list = [0.1, 1.0, 10.0, 20.0]
L_list = [3.0, 0.00000001, 100000000.0]
N = 1000
x = np.linspace(0, 100, N)
dx = x[1] - x[0]
timesteps = 10
t = np.linspace(0, timesteps -1, timesteps)
f = np.zeros(N)
dphi_dt = np.zeros(N)
f_bulk_save = np.zeros(N)
f_grad_save = np.zeros(N)
F = np.zeros(timesteps)

# implicit euler 
def run_sim(phi_0, f_0, K, L, dt):
    phi_next = np.copy(phi_0)
    for i in range(timesteps):
        f_bulk_save[1:-1] = f_0 * f_bulk(phi_0[1:-1])
        f_grad_save[1:-1] = K/2 * ((phi_0[2:] - phi_0[:-2])/(2*dx))**2
        f[1:-1] = f_0 * f_bulk(phi_0[1:-1]) + K/2 * ((phi_0[2:] - phi_0[:-2])/(2*dx))**2
        F[i] = np.trapz(f, x=x)  
        dphi_dt[1:-1] = K * div_gradient(phi_0, dx)[1:-1] - f_0 * df_bulkdx(phi_0[1:-1])
        phi_next[1:-1] = phi_0[1:-1] + L * dt * dphi_dt[1:-1]
        phi_next[0] = phi_next[1]
        phi_next[-1] = phi_next[-2]
        phi_0 = np.copy(phi_next)
    return phi_next, F, f, f_bulk_save, f_grad_save

if __name__ == '__main__':
    # plot setup 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))   
    for j in range(len(L_list)): #set index of param to vary over to [j]
        f_0 = f_0_list[0]
        K = K_list[1]
        L = L_list[j]
        dt = 0.5 * dx**2/(2*L*K)
        marks_list = ['-', '--', '-.', ':'] # different mark for every loop

        phi_0 = np.zeros(N)
        dphi_dt = np.zeros(N)
        phi_next = np.zeros(N)
        phi_0[N//2:] = 1.0 
                
        phi_next, F, f, f_bulk_save, f_grad_save = run_sim(phi_0, f_0, K, L, dt)
        
        
        #Plot 1
        ax1.plot(x, phi_next, marks_list[j] ,label='L= {}'.format(L))
        ax1.set_xlabel('x')
        ax1.set_ylabel(r'$\phi$-field')
        ax1.legend()
        #Plot 2
        ax2.plot(t, F, marks_list[j] , label='L= {}'.format(L))
        ax2.set_yscale("log")
        ax2.set_xlabel('t')
        ax2.set_ylabel('total energy F')
        ax2.legend()
        #Plot 3
        ax3.plot(x, f,marks_list[0], label='f_tot'.format(L))
        ax3.set_xlabel('X')
        ax3.set_ylabel(r'$f_{tot}$')
        ax3.legend(loc='upper left')
        
#plt.savefig('../report/graphics/')
plt.show()