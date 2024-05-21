from fipy import * 

#mesh
dx = 1
Lx = 10.
nx = 100
nx = int(Lx/dx)
mesh = Grid1D(dx=dx, nx=nx)
x = mesh.cellCenters[0]
#field varialbe
phi = CellVariable(mesh=mesh, hasOld=1)
#initial condition
phi0 = 0 
phi.setValue(value = phi0)
#boundary condtion
phi.faceValue.constrain(value=1., where=mesh.facesLeft)
phi.faceGrad.constrain(value=0., where=mesh.facesRight)
#equation coefficients
f_0 = 1. ;L =1. ; K_phi = 1.; Gamma = L*K_phi
#equation
s1 =  -2*L*f_0*phi  
eq = TransientTerm(coeff=1) == DiffusionTerm(coeff=Gamma) \
     + ImplicitSourceTerm(coeff=s1) 
#viewer
viewer = Viewer(phi)
#solver
dt = 1.
steps = 1
for step in range(steps):
     phi.updateOld()
     res = 1.e5
     while res > 1.e-3:
         res = eq.sweep(var=phi, dt=dt)
     viewer.plot()
