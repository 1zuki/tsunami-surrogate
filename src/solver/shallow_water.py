import numpy as np

class ShallowWaterSolver:
    def __init__(self, nx, ny, dx, dy, dt, g=9.81):
        # grid
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

        # time
        self.dt = dt
        self.g = g

        # vars
        self.h = np.zeros((nx, ny))   # water height
        self.hu = np.zeros((nx, ny))  # momentum in x
        self.hv = np.zeros((nx, ny))  # momentum in y

        # bathymetry
        self.b = np.zeros((nx, ny))

        self.eps = 1e-6

    def set_initial_condition(self, h0, hu0=None, hv0=None):
        """ set initial water state """
        self.h = np.where(h0 < self.eps, 0, h0)

        if hu0 is not None:
            self.hu = hu0
        
        if hv0 is not None:
            self.hv = hv0

    def set_bathymetry(self, b):
        """ set seabed height. """
        self.b = b

    def compute_velocity(self):
        """ compute velocity """
        mask = self.h > self.eps

        u = np.zeros_like(self.h)
        v = np.zeros_like(self.h)

        u[mask] = self.hu[mask] / self.h[mask]
        v[mask] = self.hv[mask] / self.h[mask]

        return u, v

    def compute_flux_x(self):
        """ compute flux F in x-direction """
        h_safe = np.maximum(self.h, self.eps)

        F = np.stack([self.hu,
                    self.hu ** 2 / h_safe + 0.5 * self.g * h_safe ** 2,
                    (self.hu * self.hv) / h_safe],
                    axis = 0)
        
        return F

    def compute_flux_y(self):
        """ compute flux G in y-direction """
        h_safe = np.maximum(self.h, self.eps)

        G = np.stack([self.hv,
                    self.hu * self.hv / h_safe,
                    self.hv ** 2 / h_safe + 0.5 * self.g * h_safe ** 2],
                    axis = 0)
        
        return G

    def compute_source(self):
        """ compute source term due to bathymetry """
        db_dx, db_dy = np.gradient(self.b, self.dx, self.dy)

        zero = np.zeros_like(self.h)

        S = np.stack([zero,
                    -self.g * self.h * db_dx,
                    -self.g * self.h * db_dy],
                    axis = 0)

        return S

    def compute_derivatives(self, F, G):
        """ compute spatial derivatives """
        dF_dx = np.zeros(F.shape)
        dG_dy = np.zeros(G.shape)

        # central
        dF_dx[:, 1:-1, :] = (F[:, 2:, :] - F[:, :-2, :]) / (2 * self.dx)
        dG_dy[:, :, 1:-1] = (G[:, :, 2:] - G[:, :, :-2]) / (2 * self.dy)
        
        # boundaries
        dF_dx[:, 0, :] = (F[:, 1, :] - F[:, 0, :]) / self.dx
        dF_dx[:, -1, :] = (F[:, -1, :] - F[:, -2, :]) / self.dx

        dG_dy[:, :, 0] = (G[:, :, 1] - G[:, :, 0]) / self.dy
        dG_dy[:, :, -1] = (G[:, :, -1] - G[:, :, -2]) / self.dy

        return dF_dx + dG_dy

    # time step update
    def update(self):
        """ one time step update """
        # flux
        F = self.compute_flux_x()
        G = self.compute_flux_y()
        divergence = self.compute_derivatives(F, G)

        # source
        S = self.compute_source()

        # current state
        U = np.stack([self.h, self.hu, self.hv], axis = 0)

        U_new = U - self.dt * divergence + self.dt * S

        self.h  = U_new[0]
        self.hu = U_new[1]
        self.hv = U_new[2]

        # update
        self.h = np.maximum(self.h, 0)

        mask = self.h < self.eps
        self.hu[mask] = 0
        self.hv[mask] = 0

    # SIMPLE VERSION WILL NEED TO CHANGE
    def apply_boundary_conditions(self):
        """ apply boundary conditions """
        # simple ver
        self.h[0, :] = self.h[1, :]
        self.h[-1, :] = self.h[-2, :]
        self.h[:, 0] = self.h[:, 1]
        self.h[:, -1] = self.h[:, -2]

        self.hu[0, :] = self.hu[1, :]
        self.hu[-1, :] = self.hu[-2, :]
        self.hu[:, 0] = self.hu[:, 1]
        self.hu[:, -1] = self.hu[:, -2]

        self.hv[0, :] = self.hv[1, :]
        self.hv[-1, :] = self.hv[-2, :]
        self.hv[:, 0] = self.hv[:, 1]
        self.hv[:, -1] = self.hv[:, -2]

    def step(self):
        """ one sim step. """
        self.adjust_dt()
        self.update()
        self.apply_boundary_conditions()

    def get_state(self):
        """ current state in tensor """
        return np.stack([self.h, self.hu, self.hv], axis=0)

    def compute_cfl(self):
        """ CFL condition for stability """
        u, v = self.compute_velocity()

        wave_speed = np.sqrt(self.g * self.h)
        speed = np.abs(u) + wave_speed

        max_speed = np.max(speed)

        cfl_x = max_speed * self.dt / self.dx
        cfl_y = max_speed * self.dt / self.dy

        return max(cfl_x, cfl_y)
        
    def adjust_dt(self, target_cfl=0.5):
        u, v = self.compute_velocity()
        wave_speed = np.sqrt(self.g * np.maximum(self.h, 0))
        speed = np.abs(u) + wave_speed

        max_speed = np.max(speed)

        if max_speed > 0:
            self.dt = target_cfl * min(self.dx, self.dy) / max_speed