#!/usr/bin/python

from sys import argv
import numpy as np


# Return next state
def next_state(s):
    if s < 5:
        s_ = 5
    else:
        self_loop = np.random.binomial(1, 0.99)
        if self_loop == 1:
            s_ = 5
        else:
            s_ = 6
    return s_


# TD(lambda) for function approximations
def temp_diff(weight, lambda_val, updates, experiment_num):
    terminal_state = 6  # Terminal State
    gamma = 0.99  # Discount Factor
    alpha = 0.001  # Learning Rate

    # States and Weights
    eligible = np.zeros(7)

    # Features of each state:
    feature_vector = np.zeros((7, 7))
    for s in range(terminal_state - 1):
        feature_vector[s, s] = 2
        feature_vector[s, terminal_state] = 1

    feature_vector[5, terminal_state] = 2
    feature_vector[5, 5] = 1

    # Experiment 1
    if experiment_num == 1:
        u = 0
        while u < updates:
            s = u % 6

            # Standard TD(0) based algorithm but its not converging, TD(0) based on Given link on assignment 4 web page
            # s_ = next_state(s)
            #
            # delta = (gamma * np.dot(feature_vector[s_, :], weight)) - np.dot(feature_vector[s, :], weight)
            # eligible = feature_vector[s, :]  # as lambda = 0
            # weight += alpha * delta * eligible
            # print ' '.join(str(e) for e in weight)

            # As episode ends when MDP enter to state 7 which is indexed as 6
            while s != 6 and u < updates:
                s_ = next_state(s)
                delta = (gamma * np.dot(feature_vector[s_, :], weight)) - np.dot(feature_vector[s, :], weight)
                eligible = feature_vector[s, :]  # as lambda = 0
                weight += alpha * delta * eligible
                s = s_
                u += 1

                # printing value function for state 1 to 6
                v_s = np.dot(feature_vector, weight)
                print ' '.join(str(e) for e in v_s[:-1])

    # Experiment 2
    elif experiment_num == 2:
        u = 0
        while u < updates:
            s = np.random.choice(np.arange(0, 5), p=[0.2, 0.2, 0.2, 0.2, 0.2])

            # As episode ends when MDP enter to state 7 which is indexed as 6
            while s != 6 and u < updates:
                s_ = next_state(s)
                delta = (gamma * np.dot(feature_vector[s_, :], weight)) - np.dot(feature_vector[s, :], weight)
                eligible = gamma * lambda_val * eligible + feature_vector[s, :]
                weight += alpha * delta * eligible
                s = s_
                u += 1

                # printing value function for state 1 to 6
                v_s = np.dot(feature_vector, weight)
                print ' '.join(str(e) for e in v_s[:-1])

    else:
        print "Please choose correct experiment i.e., 1 or 2 not", str(experiment_num)


# ############################## Input #############################
# Calling temp_diff algorithm
experiment_number = int(argv[1])
nums_update = int(argv[2])
lambda_value = float(argv[3])
# weights = np.random.rand(7)

weights = np.ones(7)
for i in range(7):
    weights[i] = float(argv[i + 4])

temp_diff(weights, lambda_value, nums_update, experiment_number)
