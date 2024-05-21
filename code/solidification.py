import numpy as np
import matplotlib.pyplot as plt

def f_density(x):
    return (x**2)*(1-x)**2

def df_density(x):
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
psi0  = np.ones(N) # create & initialize psi0 array with -1
psi0[N//2:] = 0.  # set psi0 to 1 for half of the domain

psi  =  np.zeros(N) # the array for the update


# The time loop -----------------------------------------------
for t in range(steps):
    for i in range(1,N-1):
        tmp = -K_phi*(psi0[i+1] - 2*psi0[i] + psi0[i-1])/dx**2 + f_0*df_density(psi0[i])
        psi[i] = psi0[i] - L*delta_t*tmp
        
        psi[0] = psi[1]     # Neumann condition at left
        psi[N-1] = psi[N-2] # and right boundary
    psi0 = np.copy(psi) # buffer actual time step
    plt.plot(x,psi[:],'.') # just the final time step ...
    plt.show()

    
# Plot the result ---------------------------------------------

plt.plot(x,psi[:],'.') # just the final time step ...
plt.show()
plt.show()