#!/usr/bin/env python

import matplotlib.pyplot as plt

def disc_bf_nc(p_init, p_tran, p_meas, meas, n_step):
    belief = [list(p_init)]
    n_states = len(p_init)
    for i in range(n_steps):
        belief_b = [sum([belief[i][j] * p_tran[j][k] for j in range(n_states)]) \
                for k in range(n_states)]
        belief_new = [belief_b[k] * p_meas[k][meas[i]] for k in range(n_states)]
        eta = 1 / sum(belief_new)
        belief_new = [belk * eta for belk in belief_new]
        belief.append(belief_new)
    return belief

if __name__ == '__main__':
    p_init = [0.2, 0.8, 0.0]
    p_tran = [[.7, .2, .1], [.4, .4, .2], [.2, .6, .2]]
    p_meas = [[.6, .3, .1], [.3, .6, .1], [.1, .1, 0.9]]
    n_steps = 100 
    meas = [0] * n_steps
    meas[29:31] = [2,2]
    meas[10:15] = [1,1,1,1,1]
    belief = disc_bf_nc(p_init, p_tran, p_meas, meas, n_steps)
    belief_plot = zip(*belief)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    leg = ['sunny', 'cloudy', 'rainy']
    for i in range(len(belief_plot)):
        ax.plot(range(n_steps + 1), belief_plot[i], label = leg[i], lw=2)
    ax.legend(loc='upper right')
    ax.grid('on')
    plt.show()
