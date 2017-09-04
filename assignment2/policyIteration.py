import numpy as np
from pulp import *


# ########### Reading MDP file ###################
mdp = open("data/MDP2.txt")

states = int(mdp.readline())
actions = int(mdp.readline())
print states, actions

rewards = np.zeros((states, actions, states))
for s in range(states):
    for a in range(actions):
        reward = map(float, mdp.readline().split())
        for s_ in range(states):
            rewards[s, a, s_] = reward[s_]

print rewards

transit_prob = np.zeros((states, actions, states))
for s in range(states):
    for a in range(actions):
        states_prob = map(float, mdp.readline().split())
        for s_ in range(states):
            transit_prob[s, a, s_] = states_prob[s_]

print transit_prob

discounting_factor = float(mdp.readline())
print discounting_factor

# ###################### Starting LP Code #########################

# ###################### Policy Iteration #########################


def solve_linear_equations(A, b):
    return np.linalg.solve(A, b)


A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
print solve_linear_equations(A, b)


def compute_policy_value():
    return 1


def compute_action_value():
    return 1


def find_improvable_states():
    return 1


# ##################### Howard's PI ###############################
