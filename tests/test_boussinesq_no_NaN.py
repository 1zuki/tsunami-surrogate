from src.solver.boussinesq import BoussinesqSolver
import numpy as np

nx, ny = 64, 64
solver = BoussinesqSolver(nx, ny, 1/nx, 1/ny, dt=1e-4)

b = -np.ones((nx, ny))
solver.set_bathymetry(b)

# Gaussian bump
x = np.linspace(-1,1,nx)
y = np.linspace(-1,1,ny)
X,Y = np.meshgrid(x,y,indexing="ij")
eta0 = 0.01*np.exp(-40*(X**2+Y**2))

solver.set_initial_condition(eta0)

history = solver.run(100, record_every=10, return_history=True)
print(len(history))