import numpy as np

# state indices
IX, IY, IW, IH, IVX, IVY, IVW, IVH = 0, 1, 2, 3, 4, 5, 6, 7

class Track():
    def __init__(self, id_, x, y, width, height, dt, type_: str = None):
        self.id = id_
        self.dt = dt
        self.life_count = 0
        self.unmatch_count = 0
        self.match = False
        self.tentative = True
        self.type = type_
        self.score_llr = 0     # score likelihood ration

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

    def is_moving_left(self):
        return self.KF.xs[IVY] < 0
    
    def is_moving_bottom(self):
        return self.KF.xs[IVX] > 0
    
    @property
    def score(self):
        ellr = np.exp(self.score_llr)
        return ellr/(1 + ellr)

    def __repr__(self):
        print(self.to_dict())

    def to_dict(self):
        xs = self.KF.xs
        return {
            'id': self.id,
            'top': self.bb[0].item(),
            'left': self.bb[1].item(),
            'bottom': self.bb[2].item(),
            'right': self.bb[3].item(),
            'vel_x': xs[IVX].item(),
            'vel_y': xs[IVY].item(),
            'vel_w': xs[IVW].item(),
            'vel_h': xs[IVH].item(),
            'tentative': self.tentative,
            'type': self.type,
            'score': self.score
        }

class BaseKF():
    def __init__(self):
        pass

    def _predict(self, F, Q):
        self.xs = F @ self.xs
        self.P = F @ self.P @ F.T + Q
    
    def _update(self, z, z_pred, H, R, compute_nis = False):
        y = z - z_pred
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ S_inv
        self.xs = self.xs + K @ y
        self.P = self.P - K @ H @ self.P
        if compute_nis:
            self.nis = y.T @ S_inv @ y
        return S


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

        # self.R = np.diag([5, 5, 10, 10])**2
        self.R = np.diag([1, 1, 2, 2])**2

        self.H = np.eye(self.n_z, self.n_x)

    def predict(self):
        self._predict(self.F, self.Q)

    def update(self, meas):
        z_pred = self.xs[:self.n_z]
        self.S = self._update(meas, z_pred, self.H, self.R, compute_nis=True)

