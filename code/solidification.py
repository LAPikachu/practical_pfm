import numpy as np
import matplotlib.pyplot as plt

def f_bulk(x):
    return (x**2)*(1-x)**2

def df_bulk(x):
    return 2*x*(1-x)**2 + 2*x**2 *(x-1)
# Model parameters --------------------------------------------
K_phi= 1  # gradient energy coefficient
f_0= 1  # potential energy coefficient
L = 1 # mobility parameter

# Numerical setup ---------------------------------------------
N = 101
x = np.linspace(0, 100, N)
dx = x[1]-x[0]

steps = 100
delta_t = dx**2/(2*L*K_phi) # stable time step of the FCTS scheme
print('stability limit:', delta_t)
delta_t = 0.5*delta_t
t = np.linspace(0, steps*delta_t, N)


# Initial condition -------------------------------------------
phi0  = np.ones(N) # create & initialize phi0 array with -1
phi0[N//2:] = 0.  # set phi0 to 1 for half of the domain

phi  =  np.zeros(N) # the array for the update


# The time loop -----------------------------------------------
for t in range(steps):
    for i in range(1,N-1):
        f_tot = 0.5*K_phi
        df_tot = -K_phi*(phi0[i+1] - 2*phi0[i] + phi0[i-1])/dx**2 + f_0*df_bulk(phi0[i])
        phi[i] = phi0[i] - L*delta_t*df_tot
        
        phi[0] = phi[1]     # Neumann condition at left
        phi[N-1] = phi[N-2] # and right boundary
    phi0 = np.copy(phi) # buffer actual time step
    plt.plot(x,phi[:],'.') # just the final time step ...
    plt.show()

    
# Plot the result ---------------------------------------------

plt.plot(x,phi[:],'.') # just the final time step ...
plt.show()
plt.show()