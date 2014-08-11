#!/usr/bin/env python

from random import random

def marcov_chain_simulate(n_steps, p_matrix, init_state = 0):
    p_cum_matrix = [[sum(row[0:i+1]) for i in range(len(row))] for row in p_matrix]
    states = [init_state]
    cur_state = init_state
    for i in range(n_steps):
        n_rand = random()
        for (j, p_cum) in enumerate(p_cum_matrix[cur_state]):
            if n_rand < p_cum:
                next_state = j
                break
        else:
            next_state = len(p_cum_matrix[cur_state]) - 1
        cur_state = next_state
        states.append(cur_state)
    return states

if __name__ == '__main__':
    n_steps = 1000000
    p_matrix = [[0.8, 0.2, 0.0], [0.4, 0.4, 0.2], [0.2, 0.6, 0.2]]
    states = marcov_chain_simulate(n_steps, p_matrix)
    stat_prob = [float(states.count(i))/len(states) for i in sorted(set(states))]
    print stat_prob
