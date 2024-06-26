import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def grad_2D(c0, dx):
    tmp = np.copy(c0)
    tmp[1:-1, 1:-1] = (c0[1:-1,:-2] - c0[1:-1,2:])/(dx)
    return tmp

def div_grad_2D(c0, dx): 
    tmp = np.copy(c0)
    tmp[1:-1, 1:-1] = (c0[1:-1,:-2] + c0[1:-1,2:] - 4*c0[1:-1,1:-1] +c0[:-2,1:-1] + c0[2:,1:-1])/dx**2
    return tmp


def dbulk_dx(x, c1, c2):
    return (x - c2)* (x - c1)* (-2*x + c1 + c2) 

f_0_mol = 134e6
V_mol = 1e-5 
f_0 = f_0_mol 
num_K = 10
M = 1.0e-17 # mol^2 / Jms * (m**3/mol)
dt = 0.1
dx = 1.0e-8
N = 100
c1 = 0.16
c2 = 0.23
K = 1*10**-6
x = np.linspace(0, 1.0e-8*N, N, dtype=np.float64)
y = np.linspace(0, 1.0e-8*N, N, dtype=np.float64)
timesteps = 50000 
t = np.linspace(0, timesteps -1, timesteps)
dc_dt =  np.zeros((N,N), dtype=np.float64)
tmp = np.zeros((N,N), dtype=np.float64)
tmp2 = np.zeros((N,N), dtype=np.float64)
tmp3 = np.zeros((N,N), dtype=np.float64)
def run_sim( f_0, K, M, dt, c1, c2):
    c_save_list = []
    t_save_list = []
    c0 = np.random.normal(loc=0.195, scale=0.01, size=(N,N)) 
    c_next = np.zeros((N,N), dtype=np.float64)
    for j in range(timesteps):
        tmp2 =  - K * div_grad_2D(c0, dx) - 2*(f_0/V_mol) * dbulk_dx(c0 ,c1, c2) 
        dc_dt = M* (V_mol**2) *div_grad_2D(tmp2, dx)
        c_next = c0 + dt * dc_dt
        c_next[0,:] = c_next[1,:]
        c_next[N-1,:] = c_next[N-2,:]
        c_next[:,0] = c_next[:,1]
        c_next[:,N-1] = c_next[:,N-2]
        c0 = np.copy(c_next)
        if j == 0 or j%(timesteps//4) == 0:
            c_save_list.append(c0)
            t_save_list.append(j*dt)
    return c_save_list, t_save_list

if __name__ == '__main__':
    c_save_list, t_save_list = run_sim(f_0 = f_0, K = K, M=M, dt=dt, c1=c1, c2=c2)

    # Plotting
    vmin = c1-0.1
    vmax = c2+0.1
    fig, axs = plt.subplots(ncols=5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.05]}, figsize=(40, 10))
    sns.heatmap(c_save_list[0], ax=axs[0], cbar=False, vmin=vmin, vmax=vmax) 
    sns.heatmap(c_save_list[1], ax=axs[1], cbar=False, vmin=vmin, vmax=vmax) 
    sns.heatmap(c_save_list[2], ax=axs[2], cbar=False, vmin=vmin, vmax=vmax) 
    sns.heatmap(c_save_list[3], ax=axs[3], cbar=False, vmin=vmin, vmax=vmax) 
    for i,ax in enumerate(axs[:-1]):
        ax.axis('off')
        ax.set_title('t = {:3.2g} s'.format(t_save_list[i]))
    fig.colorbar(axs[1].collections[0], cax=axs[4])
    axs[4].set_ylabel('Concentration')

    #fig.savefig('../report/graphics/NiAl_2D_K.png')
    plt.show()
    print("cutoff prevention")