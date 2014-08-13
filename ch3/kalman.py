#!/usr/bin/env python
#
''' Roughly corresponds to problems 1 and 2 from chapter #3.

Implements Kalman filter with optional measurement update step.
No contour plots yet in the testing. 
'''

import numpy as np

class KalmanFilter:
    def __init__(self, A, B, C, R, Q):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
    
    def filter(self, mu_tm1, sigma_tm1, u_t, z_t, meas_update):
        '''Implements one filtering step of Kalman filter.

        mu_tm1, sigma_tm1 - mean and sigma from the step t-1
        u_t, z_t - control and measurement
        meas_update - perform measurement update if meas_update is 'yes'
        '''
        mu_t_bar = (self.A).dot(mu_tm1) + (self.B).dot(u_t)
        sigma_t_bar = (self.A).dot(sigma_tm1).dot(self.A.T) + R
        if meas_update is 'y' or meas_update is 'yes':
            ms = (self.C).dot(sigma_t_bar).dot(self.C.T) + Q
            msi = np.linalg.inv(ms)
            K = (sigma_t_bar).dot(self.C.T).dot(msi)
            mu_t = mu_t_bar + (K).dot((z_t - self.C.dot(mu_t_bar)))
            sigma_t = (np.eye(len(mu_t)) - (K).dot(self.C)).dot(sigma_t_bar)
            return [mu_t, sigma_t]
        else:
            return [mu_t_bar, sigma_t_bar]

if __name__ == '__main__':
    A = np.array([1., 1., 0., 1.]).reshape(2, 2)
    B = np.array([.5, 1.]).reshape(2, 1)
    R = np.array([0.25, 0.5, 0.5, 1]).reshape(2, 2)
    C = np.array([[1., 0.]])
    Q = np.array([[10]])
    kf = KalmanFilter(A, B, C, R, Q)
    n_steps = 5
    u_t = np.array([0] * n_steps)
    z_t = np.array([5] * n_steps)
    meas_update = ['n'] * (n_steps - 1) + ['y']
    state = [[np.array([[0., 0.]]).T, np.array([[0., 0.], [0., 0.]])]]
    for i in range(n_steps):
        state.append(kf.filter(state[i][0], state[i][1], u_t[i], z_t[i], meas_update[i]))
        print state[-1][0]
        print state[-1][1]
