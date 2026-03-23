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
        h0 = np.asarray(h0, dtype=float)

        if h0.shape != (self.nx, self.ny):
            raise ValueError(f"h0 shape must be {(self.nx, self.ny)}, got {h0.shape}")

        self.h = h0.copy()
        self.h[self.h < 0] = 0.0

        self.hu = np.zeros_like(self.h)
        self.hv = np.zeros_like(self.h)

        if hu0 is not None:
            hu0 = np.asarray(hu0, dtype=float)
            
            if hu0.shape != self.h.shape:
                raise ValueError("hu0 shape mismatch")
            
            self.hu = hu0.copy()

        if hv0 is not None:
            hv0 = np.asarray(hv0, dtype=float)
            
            if hv0.shape != self.h.shape:
                raise ValueError("hv0 shape mismatch")

            self.hv = hv0.copy()

        mask = self.h < self.eps
        self.hu[mask] = 0.0
        self.hv[mask] = 0.0

    def set_bathymetry(self, b):
        b = np.asarray(b, dtype=float)

        if b.shape != (self.nx, self.ny):
            raise ValueError(f"b shape must be {(self.nx, self.ny)}, got {b.shape}")

        self.b = b.copy()

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

    def compute_rusanov_flux_F(self, F):
        U = np.stack([self.h, self.hu, self.hv], axis = 0)

        F_half = np.zeros_like(F)

        U_L = U[:, :-1, :]
        U_R = U[:, 1:, :]
    
        F_L = F[:, :-1, :]
        F_R = F[:, 1:, :]

        u, v = self.compute_velocity()
        c = np.sqrt(self.g * np.maximum(self.h, self.eps))

        lbd_L = np.abs(u[:-1, :]) + c[:-1, :]
        lbd_R = np.abs(u[1:, :]) + c[1:, :]

        lbd = np.maximum(lbd_L, lbd_R)
        lbd = lbd[None, :, :] # broadcasting

        F_half[:, :-1, :] = 0.5 * (F_L + F_R) - 0.5 * lbd * (U_R - U_L)

        dF_dx = np.zeros_like(F)

        dF_dx[:, 1:-1, :] = (F_half[:, 1:-1, :] - F_half[:, 0:-2, :]) / self.dx

        dF_dx[:, 0, :] = dF_dx[:, 1, :]
        dF_dx[:, -1, :] = dF_dx[:, -2, :]

        return dF_dx
    
    def compute_rusanov_flux_G(self, G):
        U = np.stack([self.h, self.hu, self.hv], axis = 0)

        G_half = np.zeros_like(G)

        U_L = U[:, :, :-1]
        U_R = U[:, :, 1:]

        G_L = G[:, :, :-1]
        G_R = G[:, :, 1:]

        u, v = self.compute_velocity()
        c = np.sqrt(self.g * np.maximum(self.h, self.eps))

        lbd_L = np.abs(v[:, :-1]) + c[:, :-1]
        lbd_R = np.abs(v[:, 1:]) + c[:, 1:]

        lbd = np.maximum(lbd_L, lbd_R)
        lbd = lbd[None, :, :]

        G_half[:, :, :-1] = 0.5 * (G_L + G_R) - 0.5 * lbd * (U_R - U_L)

        dG_dy = np.zeros_like(G)

        dG_dy[:, :, 1:-1] = (G_half[:, :, 1:-1] - G_half[:, :, 0:-2]) / self.dy

        dG_dy[:, :, 0] = dG_dy[:, :, 1]
        dG_dy[:, :, -1] = dG_dy[:, :, -2]

        return dG_dy

    # time step update
    def update(self):
        """ one time step update """
        # flux
        F = self.compute_flux_x()
        G = self.compute_flux_y()

        dF_dx = self.compute_rusanov_flux_F(F)
        dG_dy = self.compute_rusanov_flux_G(G)

        divergence = dF_dx + dG_dy

        # source
        S = self.compute_source()

        # current state
        U = np.stack([self.h, self.hu, self.hv], axis = 0)

        U_new = U - self.dt * divergence + self.dt * S

        self.h  = np.maximum(U_new[0], 0.0)
        self.hu = U_new[1]
        self.hv = U_new[2]

        # update
        mask = self.h < self.eps

        self.hu[mask] = 0
        self.hv[mask] = 0

    # non refective boundary (open ocen tsunami)
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
        self.apply_boundary_conditions()
        self.update()
        self.apply_boundary_conditions()

    def get_state(self):
        """ current state in tensor """
        return np.stack([self.h, self.hu, self.hv], axis=0)

    def compute_cfl(self):
        """ CFL condition for stability """
        u, v = self.compute_velocity()

        wave_speed = np.sqrt(self.g * np.maximum(self.h, self.eps))
        speed_x = np.abs(u) + wave_speed
        speed_y = np.abs(v) + wave_speed

        max_speed = max(np.max(speed_x), np.max(speed_y))

        cfl_x = max_speed * self.dt / self.dx
        cfl_y = max_speed * self.dt / self.dy

        return max(cfl_x, cfl_y)
        
    def adjust_dt(self, target_cfl=0.5):
        u, v = self.compute_velocity()
        wave_speed = np.sqrt(self.g * np.maximum(self.h, self.eps))
        
        speed_x = np.abs(u) + wave_speed
        speed_y = np.abs(v) + wave_speed

        max_speed = max(np.max(speed_x), np.max(speed_y))

        if max_speed > 0:
            self.dt = target_cfl * min(self.dx, self.dy) / max_speed


"""
Reference doc:
https://www.sciencedirect.com/science/article/pii/S0307904X04001647
https://en.wikipedia.org/wiki/Shallow_water_equations
https://www.sciencedirect.com/science/article/pii/S0045793026000423
"""