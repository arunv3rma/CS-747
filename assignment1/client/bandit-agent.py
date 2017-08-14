#!/usr/bin/env python
import numpy as np
from scipy.optimize import bisect
from scipy.stats import entropy
import socket
import sys


def epsilon_greedy(epsilon, numArms, arm_means):
    e_step = np.random.binomial(1, epsilon)
    # print e_step
    if e_step == 1:
        selected_arm = int(np.random.uniform(0, numArms, 1))
    else:
        # print arm_means
        selected_arm = np.argmax(arm_means)
    return selected_arm


def ucb(arm_means, num_pulls):
    # arms_upper_bound = np.zeros(numArms)
    selected_arm = np.argmax(arm_means + np.sqrt((3 * np.log(np.sum(num_pulls))) / (2 * num_pulls)))
    return selected_arm


# Limit checking function
def check_condition(bound, arm_expected_reward, sample_count, log_val):
    p = [arm_expected_reward, 1 - arm_expected_reward]
    q = [bound, 1 - bound]
    return sample_count * entropy(p, qk=q) - log_val


# Computing Upper and Lower bound
def upper_bound(arm_expected_reward, sample_count, log_val):
    try:
        upper_bound_val = bisect(check_condition, arm_expected_reward, 1,
                                 args=(arm_expected_reward, sample_count, log_val))
    except:
        upper_bound_val = 1
    return upper_bound_val


def kl_ucb(numArms, arm_rewards, arm_pulls):
    arms_upper_bound = np.zeros(numArms)
    log_val = np.log(np.sum(arm_pulls)) + 3 * np.log(np.log(np.sum(arm_pulls)))
    for arm in range(numArms):
        arms_upper_bound[arm] = upper_bound(arm_rewards[arm] / arm_pulls[arm], arm_pulls[arm], log_val)
    selected_arm = np.argmax(arms_upper_bound)
    return selected_arm


def ts(numArms, arms_success, arms_failure):
    arms_prob = np.zeros(numArms)
    for arm in range(numArms):
        arms_prob[arm] = np.random.beta(arms_success[arm], arms_failure[arm])
    selected_arm = np.argmax(arms_prob)
    return selected_arm


# print command line arguments
numArms = int(sys.argv[2])
randomSeed = float(sys.argv[4])
horizon = int(sys.argv[6])
hostname = sys.argv[8]
port = int(sys.argv[10])
algorithm = sys.argv[12]
epsilon = float(sys.argv[14])

# np.random.seed(randomSeed)

# print numArms, randomSeed, horizon, hostname, port, algorithm, epsilon

# TCP connection details
BUFFER_SIZE = 256
server_address = (hostname, port)

#  Socket Connection with server
socket_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
socket_connection.connect(server_address)

if algorithm == "epsilon-greedy":
    arm_rewards = np.zeros(numArms)
    arm_pulls = np.ones(numArms)
    arm_means = np.zeros(numArms)

    for i in range(horizon):
        # Getting rewards for given arm from environment
        arm = epsilon_greedy(epsilon, numArms, arm_means)
        socket_connection.send(str(arm))
        data = socket_connection.recv(BUFFER_SIZE)

        # if data == '':
        #     i -= 1
        # else:

        # Updating means of arms
        arm_rewards[arm] += int(data.split(",")[0])
        arm_pulls[arm] += 1
        arm_means[arm] = arm_rewards[arm] / arm_pulls[arm]


elif algorithm == "UCB":
    arm_rewards = np.zeros(numArms)
    arm_pulls = np.ones(numArms)
    arm_means = np.zeros(numArms)

    for i in range(horizon):
        # Getting rewards for given arm from environment
        arm = ucb(arm_means, arm_pulls)
        socket_connection.send(str(arm))
        data = socket_connection.recv(BUFFER_SIZE)
        # print data

        # Updating means of arms
        arm_rewards[arm] += int(data.split(",")[0])
        arm_pulls[arm] += 1
        arm_means[arm] = arm_rewards[arm] / arm_pulls[arm]

elif algorithm == "KL-UCB":
    arm_rewards = np.zeros(numArms)
    arm_pulls = np.zeros(numArms)

    for arm in range(numArms):
        socket_connection.send(str(arm))
        data = socket_connection.recv(BUFFER_SIZE)

        # Updating means of arms
        arm_rewards[arm] += int(data.split(",")[0])
        arm_pulls[arm] += 1

    for i in range(numArms + 1, horizon):
        # Getting rewards for given arm from environment
        arm = kl_ucb(numArms, arm_rewards, arm_pulls)
        socket_connection.send(str(arm))
        data = socket_connection.recv(BUFFER_SIZE)
        # print data

        # Updating means of arms
        arm_rewards[arm] += int(data.split(",")[0])
        arm_pulls[arm] += 1

elif algorithm == "Thompson-Sampling":
    arms_success = np.ones(numArms)
    arms_failure = np.ones(numArms)

    for i in range(horizon):
        # Getting rewards for given arm from environment
        arm = ts(numArms, arms_success, arms_failure)
        socket_connection.send(str(arm))
        data = socket_connection.recv(BUFFER_SIZE)
        # print data

        # Updating means of arms
        success = int(data.split(",")[0])
        arms_success[arm] += success
        arms_failure[arm] += 1 - success

    # print arms_success, arms_failure

else:
    # rr return best performing algorithm
    arms_success = np.ones(numArms)
    arms_failure = np.ones(numArms)

    for i in range(horizon):
        # Getting rewards for given arm from environment
        arm = ts(numArms, arms_success, arms_failure)
        socket_connection.send(str(arm))
        data = socket_connection.recv(BUFFER_SIZE)
        # print data

        # Updating means of arms
        success = int(data.split(",")[0])
        arms_success[arm] += success
        arms_failure[arm] += 1 - success

socket_connection.close()
