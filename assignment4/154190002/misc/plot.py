#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


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
def temp_diff(weight, updates, experiment_num):
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

    values = []

    # Experiment 1
    if experiment_num == 1:
        u = 0
        while u < updates:
            s = u % 6

            # As episode ends when MDP enter to state 7 which is indexed as 6
            while s != 6 and u < updates:
                s_ = next_state(s)
                delta = (gamma * np.dot(feature_vector[s_, :], weight)) - np.dot(feature_vector[s, :], weight)
                eligible = feature_vector[s, :]  # as lambda = 0
                weight += alpha * delta * eligible
                v_s = np.dot(feature_vector, weight)
                values.append(list(v_s[:-1]))
                s = s_
                u += 1

        # Plotting plot1 for experiment1
        vs = np.array(values)
        x = range(1, updates + 1)
        z = np.zeros(updates)
        shape = ['--*', '--+', '--H', '--d', '--X', '--^', '--v']

        for i in range(terminal_state):
            plt.plot(x, vs[:, i], shape[i], label='v(s' + str(i + 1) + ')')

        plt.plot(x, z, color='k', linewidth=2.0)
        plt.legend(loc='upper right', numpoints=1)
        plt.title("Estimated Value of State v/s Number of Updates")
        plt.ylabel("Estimated Value of State")
        plt.xlabel("Number of Updates")

        plt.savefig("plot" + str(experiment_num) + ".png", bbox_inches='tight')
        plt.close()

    # Experiment 2
    elif experiment_num == 2:
        u = 0
        lambda_val = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        while u < updates:
            s = np.random.choice(np.arange(0, 5), p=[0.2, 0.2, 0.2, 0.2, 0.2])

            # As episode ends when MDP enter to state 7 which is indexed as 6
            while s != 6 and u < updates:
                s_ = next_state(s)
                delta = (gamma * np.dot(feature_vector[s_, :], weight)) - np.dot(feature_vector[s, :], weight)

                avg_vs = []
                for l in lambda_val:
                    eligible = gamma * l * eligible + feature_vector[s, :]
                    weight += alpha * delta * eligible
                    avg_vs.append(np.mean(list(np.dot(feature_vector, weight))[:-1]))

                values.append(avg_vs)
                s = s_
                u += 1

        # Plotting plot1 for experiment1
        vs = np.array(values)
        x = range(1, updates + 1)
        z = np.zeros(updates)
        shape = ['--*', '--+', '--H', '--d', '--X', '--^', '--v']

        for i in range(len(lambda_val)):
            plt.plot(x, vs[:, i], shape[i], label='lambda = ' + str(lambda_val[i]))

        plt.plot(x, z, color='k', linewidth=2.0)
        plt.legend(loc='upper right', numpoints=1)
        plt.title("Estimated Average Value of States v/s Number of Updates")
        plt.ylabel("Estimated Average Value of States")
        plt.xlabel("Number of Updates")

        plt.savefig("plot" + str(experiment_num) + ".png", bbox_inches='tight')
        plt.close()

    else:
        print "Please choose correct experiment i.e., 1 or 2 not", str(experiment_num)

# ############################## Input #############################
# Calling temp_diff algorithm
experiment_number = 2
nums_update = 1000000
weights = np.ones(7)

temp_diff(weights, nums_update, experiment_number)
