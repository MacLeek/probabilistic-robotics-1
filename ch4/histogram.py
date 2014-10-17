#!/usr/bin/env python
#
''' Roughly corresponds to problems 1 and 2 from chapter #4.

Implements histogram filter with optional measurement update step.

Not thoroughly tested.
'''

import numpy as np
import matplotlib.pyplot as plt

class HistogramFilter:
    def __init__(self, A, B, C, R, Q, xmin, xmax, nx, vmin, vmax, nv):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        # for simplicity, use static fixed-size grid with nx x nv elements
        self.nx = nx
        self.nv = nv
        self.x = np.linspace(xmin, xmax, nx)
        self.v = np.linspace(vmin, vmax, nv)
        self.belief = np.zeros((nx, nv))
        self.belief[nx / 2, nv / 2] = 1     # this shoud be (0, 0) point, we are certain of
                                            # initial location
    
    def p_tran(self, x_t, x_tm1, u_t):
        f1 = np.linalg.det(2 * np.pi * self.R) ** (-0.5)
        f2 = x_t - (self.A).dot(x_tm1) - (self.B).dot(u_t)  
        ex = -0.5 * (f2.T).dot(np.linalg.inv(self.R)).dot(f2)
        return f1 * np.exp(ex)

    def p_meas(self, z_t, x_t):
        f1 = np.linalg.det(2 * np.pi * self.Q) ** (-0.5)
        f2 = z_t - (C).dot(x_t)
        ex = -0.5 * (f2.T).dot(np.linalg.inv(self.Q)).dot(f2)
        return f1 * np.exp(ex)

    
    def filter(self, u_t, z_t, meas_update):
        ''' Implements one filtering step of histogram filter.

        u_t, z_t - control and measurement
        meas_update - perform measurement update if meas_update is 'yes'
        '''
        belief_old = np.copy(self.belief)
        for i in range(self.nx):
            for j in range(self.nv):
                x_t = np.vstack([self.x[i], self.v[j]])
                self.belief[i, j] = 0
                xt_size = (self.x[1] - self.x[0]) * (self.v[1] - self.v[0])
                for k in range(self.nx):
                    for l in range(self.nv):
                        x_tm1 = np.vstack([self.x[k], self.v[l]])
                        self.belief[i, j] += belief_old[k, l] * self.p_tran(x_t, x_tm1, u_t)\
                                * xt_size

        if meas_update is 'y' or meas_update is 'yes':
            for i in range(self.nx):
                for j in range(self.nv):
                    x_t = np.vstack([self.x[i], self.v[j]])
                    self.belief[i, j] *= self.p_meas(z_t, x_t)
        self.belief = self.belief / sum(sum(self.belief))
    def get_belief(self):
        return self.belief

if __name__ == '__main__':
    A = np.array([1., 1., 0., 1.]).reshape(2, 2)
    B = np.array([0., 0.]).reshape(2, 1)
    R = np.array([0.25, 0.5, 0.5, 1.1]).reshape(2, 2)
    C = np.array([[1., 0.]])
    Q = np.array([[10]])
    (xmin, xmax, nx) = (-5., 5., 23)
    (vmin, vmax, nv) = (-5., 5., 23)
    hf = HistogramFilter(A, B, C, R, Q, xmin, xmax, nx, vmin, vmax, nv)
    n_steps = 3
    u_t = np.zeros((n_steps,))
    z_t = np.ones((n_steps,)) * 3
    meas_update = ['y'] * n_steps
    for i in range(n_steps):
        hf.filter(u_t[i], z_t[i], meas_update[i])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(vmin, vmax, nv))
        z = hf.get_belief()
        plt.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()))
        plt.colorbar()
    plt.show()
