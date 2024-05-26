import numpy as np

#

def f_bulk(x):
    return x**2 * (1 - x)**2

def df_bulk_dx(x):
    return 2*x - 6*x**2 + 4*x**3

def lapla_1D(mat, dx):
    tmp = np.copy(mat)
    tmp[1:-1] = (mat[:-2] + mat[2:] - 2 * mat[1:-1]) / dx**2
    return tmp

def lapla_2D(c0, dx):
    tmp = np.copy(c0)
    tmp[1:-1, 1:-1] = (c0[1:-1,:-2] + c0[1:-1,2:] - 4*c0[1:-1,1:-1] +c0[:-2,1:-1] + c0[2:,1:-1])/dx**2
    return tmp

def bulk_NiAl(x):
    return (0.16 - x)**2 * (x - 0.23)**2
def dbulk_dx(x):
    return 4.0*x**3 - 1.17*x**2 + 0.23*x - 0.014
def d2bulk_dx(x):
    return 2 * (0.39 - 2*x)**2 - 4 * (0.16 - x) * (x - 0.23)
def d3bulk_dx(x):
    return -12 * (0.39 - 2 * x)

f_0_mol = 134e6
V_mol = 1e-5 # mol^2 / Jms
f_0 = f_0_mol * V_mol 
num_K = 100
K = np.linspace(1e-7, 1e-5, num_K, dtype=np.float64)
M = 1.0e-17 * (V_mol**2)
dt = 1
dx = 1.0e-8
N = 100
x = np.linspace(0, 1.0e-8*N, N, dtype=np.float64)
timesteps = 1000
F = np.zeros(num_K, dtype=np.float64)
F_bulk = np.zeros(num_K, dtype=np.float64)
F_grad = np.zeros(num_K, dtype=np.float64)
dc_dt = np.zeros(N, dtype=np.float64)
tmp = np.zeros(N, dtype=np.float64)

# Initialisierung:
c0 = np.zeros(N, dtype=np.float64)
c_next = np.zeros(N, dtype=np.float64)
c0[:N//2] = 0.16
c0[N//2:] = 0.23

for i in range(num_K):
    c0[:N//2] = 0.16
    c0[N//2:] = 0.23
    for j in range(timesteps):
        f_grad =  0.5*K[i]*lapla_1D(c0, dx)[1:-1]
        f_bulk = f_0 * dbulk_dx(c0[1:-1])
        f_inter = f_grad + f_bulk
        tmp[1:-1] = -1*f_grad + f_bulk
        c_next[1:-1] = c0[1:-1] + lapla_1D(tmp, dx)[1:-1] * dt * M
        c0 = np.copy(c_next)
    F_grad[i] = np.trapz(f_grad, x=x[1:-1])
    F_bulk[i] = np.trapz(f_bulk, x=x[1:-1])
    F[i] = np.trapz(f_inter, x=x[1:-1])
print(F)