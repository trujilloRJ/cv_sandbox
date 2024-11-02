import numpy as np

# state indices
IX, IY, IW, IH, IVX, IVY, IVW, IVH = 0, 1, 2, 3, 4, 5, 6, 7

class Track():
    def __init__(self, id_, x, y, width, height, dt):
        self.id = id_
        self.dt = dt
        self.life_count = 0
        self.unmatch_count = 0
        self.match = False
        self.tentative = True

        self.KF = LKF_CV(x, y, width, height, dt)

        self.update_bounding_box()

    def predict(self):
        self.KF.predict()
        self.update_bounding_box()

    def update(self, meas: np.ndarray):
        self.KF.update(meas)
        self.update_bounding_box()

    def update_bounding_box(self):
        # top, left, bottom, right
        xs = self.KF.xs
        self.bb = np.array([
            int(xs[IX] - xs[IH] // 2),
            int(xs[IY] - xs[IW] // 2),
            int(xs[IX] + xs[IH] // 2),
            int(xs[IY] + xs[IW] // 2)
        ])


class BaseKF():
    def __init__(self):
        pass

    def _predict(self, F, Q):
        self.xs = F @ self.xs
        self.P = F @ self.P @ F.T + Q
    
    def _update(self, z, z_pred, H, R):
        y = z - z_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.xs = self.xs + K @ y
        self.P = self.P - K @ H @ self.P


class LKF_CV(BaseKF):
    def __init__(self, x, y, w, h, dt):
        self.n_x = 8
        self.n_z = 4
        self.xs = np.array([x, y, w, h, 0, 0, 0, 0])
        self.P = np.diag([20, 20, 20, 20, 20, 20, 20, 20])**2

        self.F = np.eye(self.n_x)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt
        self.F[3, 7] = dt

        # TODO: check for a better Q
        self.Q = np.diag([2] * self.n_x)**2

        self.R = np.diag([5, 5, 10, 10])**2

        self.H = np.eye(self.n_z, self.n_x)

    def predict(self):
        self._predict(self.F, self.Q)

    def update(self, meas):
        z_pred = self.xs[:self.n_z]
        self._update(meas, z_pred, self.H, self.R)

