import numpy as np
import matplotlib.pyplot as plt

def grad_1D(x, dx):
    tmp = np.copy(x)
    tmp[1:-1] = x[2:] - x[:-2] / dx
    return tmp

def div_grad_1D(x, dx):
    tmp = np.copy(x)
    tmp[1:-1] = (x[:-2] + x[2:] - 2 * x[1:-1]) / dx**2
    return tmp

def bulk_NiAl(x,c1, c2):
    return (c2 - x)**2 * (x - c1)**2

def dbulk_dx(x, c1, c2):
    return ((c2 - x) *(x- c1)**2) +((x - c1)*(c2 - x)**2) 

f_0_mol = 134e6
V_mol = 1e-5 
f_0 = f_0_mol 
num_K = 10
K_list = np.linspace(1e-5, 3e-7, num_K, dtype=np.float64)
M = 1.0e-17 * (V_mol**2)# mol^2 / Jms * (m**3/mol)
dt = 0.1
dx = 1.0e-8
N = 100
c1 = 0.165
c2 = 0.23
x = np.linspace(0, 1.0e-8*N, N, dtype=np.float64)
timesteps = 3000
t = np.linspace(0, timesteps -1, timesteps)
F = np.zeros(timesteps, dtype=np.float64)
dc_dt =  np.zeros(N, dtype=np.float64)
f_inter = np.zeros(N, dtype=np.float64)

def run_sim( f_0, K, M, dt, c1, c2):
    c0 = np.zeros(N, dtype=np.float64)
    c_next = np.zeros(N, dtype=np.float64)
    c0[:N//2] = c1
    c0[N//2:] = c2
    for j in range(timesteps):
        f_inter[1:-1] = f_0 * bulk_NiAl(c0[1:-1],c1, c2) + K/2 * ((c0[2:] - c0[:-2])/(2*dx))**2
        dc_dt[1:-1] = f_0 * dbulk_dx(c0[1:-1],c1, c2) - K * div_grad_1D(c0, dx)[1:-1]
        F[j]=np.trapz(f_inter, x=x)
        c_next[1:-1] = c0[1:-1] + div_grad_1D(dc_dt, dx)[1:-1] * dt * M
        c_next[0] = c_next[1]
        c_next[-1] = c_next[-2]
        c0 = np.copy(c_next)
    return c_next, F, f_inter

if __name__ == '__main__':
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5)) 
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.constrained_layout.use'] = True
    for K in K_list:
        c_next, F, f_inter = run_sim(f_0 = f_0, K = K, M=M, dt=dt, c1=c1, c2=c2) 
        print('K = {K} F = {F}'.format(K=K, F=F[-1]))
        ax1.plot(x, c_next, label = 'K= {}'.format(K))
        ax1.legend()
        ax1.set_xlabel('x')
        ax1.set_ylabel('c')
        ax2.plot(t, F, label = 'K= {}'.format(K))
        ax2.set_yscale('log')
        ax2.legend()
        ax2.set_xlabel('t in s')
        ax2.set_ylabel(r'F in $\frac{J}{m^{2}}$')
        ax3.plot(x, f_inter, label = 'K= {}'.format(K))
        ax3.legend()
        ax3.set_xlabel('x')
        ax3.set_ylabel(r'f in $\frac{J}{mol}$')
    plt.savefig('report/graphics/find_K_grad_energy_density.png')
    plt.show()
    print("cutoff prevention")