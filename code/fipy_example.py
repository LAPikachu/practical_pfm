from fipy import * 

#mesh
dx = 0.1
Lx = 10.
nx = 100
nx = int(Lx/dx)
mesh = Grid1D(dx=dx, nx=nx)
x = mesh.cellCenters[0]
#field varialbe
phi = CellVariable(mesh=mesh, hasOld=1)
#initial condition
phi0 = GaussianNoiseVariable(mesh=mesh, mean=0.0, variance=1.0, hasOld=0)
phi.setValue(value = phi0.value)
#boundary condtion
phi.faceValue.constrain(value=0., where=mesh.facesLeft)
phi.faceGrad.constrain(value=1., where=mesh.facesRight)
#equation coefficients
rho = 1. ; u = (20.,) ; Gamma = 5. ; a = 1. ; b = 1. ; c = 1.
#equation
s0 = c
s1 = a * phi + b
eq = TransientTerm(coeff=rho) == - ConvectionTerm(coeff=u) \
     + DiffusionTerm(coeff=Gamma) \
     + ImplicitSourceTerm(coeff=s1) \
     + s0
#viewer
viewer = Viewer(phi)
#solver
dt = 1.e-2
steps = 100
for step in range(steps):
     phi.updateOld()
     res = 1.e5
     while res > 1.e-3:
         res = eq.sweep(var=phi, dt=dt)
     viewer.plot()
