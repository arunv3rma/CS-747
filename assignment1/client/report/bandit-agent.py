#!/usr/bin/env python
import numpy as np
from scipy.optimize import bisect
from scipy.stats import entropy
import socket
import sys
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


def get_rewards(arm_true_means, arm):
    return np.random.binomial(1, arm_true_means[arm])


# Parameters
algorithm = sys.argv[1]
# algorithm = "epsilon-greedy"
# algorithm = "UCB"
# algorithm = "KL-UCB"
# algorithm = "Thompson-Samplingy"
horizons = [10, 100, 1000, 10000, 100000]
nRuns = int(sys.argv[2])
epsilon = float(sys.argv[3])

if int(sys.argv[4]) == 0:
    arm_true_means = [0.1, 0.2, 0.3, 0.4, 0.5]
else:
    arm_true_means = [0.12, 0.14, 0.19, 0.21, 0.24, 0.24, 0.25, 0.27, 0.30, 0.31, 0.33, 0.36, 0.38, 0.40, 0.41, 0.46, 0.53, 0.56, 0.61, 0.71, 0.86, 0.87, 0.90, 0.92, 0.97]

numArms = len(arm_true_means)


regret_horizon = []

for horizon in horizons:
    max_reward = horizon * max(arm_true_means)

    if algorithm == "epsilon-greedy":
        regret = []

        for n in range(nRuns):
            arm_rewards = np.zeros(numArms)
            arm_pulls = np.ones(numArms)
            arm_means = np.zeros(numArms)
            # np.random.seed(int(np.random.uniform(1, 100, 1)))

            for i in range(horizon):
                # Getting rewards for given arm from environment
                arm = epsilon_greedy(epsilon, numArms, arm_means)

                # Updating means of arms
                arm_rewards[arm] += get_rewards(arm_true_means, arm)
                arm_pulls[arm] += 1
                arm_means[arm] = arm_rewards[arm] / arm_pulls[arm]
            regret.append(max_reward - np.sum(arm_rewards))

    elif algorithm == "UCB":
        regret = []

        for n in range(nRuns):
            arm_rewards = np.zeros(numArms)
            arm_pulls = np.ones(numArms)
            arm_means = np.zeros(numArms)

            for i in range(horizon):
                # Getting rewards for given arm from environment
                arm = ucb(arm_means, arm_pulls)

                # Updating means of arms
                arm_rewards[arm] += get_rewards(arm_true_means, arm)
                arm_pulls[arm] += 1
                arm_means[arm] = arm_rewards[arm] / arm_pulls[arm]

            regret.append(max_reward - np.sum(arm_rewards))

    elif algorithm == "KL-UCB":
        regret = []

        for n in range(nRuns):
            arm_rewards = np.zeros(numArms)
            arm_pulls = np.ones(numArms)

            for arm in range(numArms):
                # Updating means of arms
                arm_rewards[arm] += get_rewards(arm_true_means, arm)
                arm_pulls[arm] += 1

            for i in range(numArms + 1, horizon):
                # Getting rewards for given arm from environment
                arm = kl_ucb(numArms, arm_rewards, arm_pulls)

                # Updating means of arms
                arm_rewards[arm] += get_rewards(arm_true_means, arm)
                arm_pulls[arm] += 1

            regret.append(max_reward - np.sum(arm_rewards))

    elif algorithm == "Thompson-Sampling":
        regret = []

        for n in range(nRuns):
            arm_rewards = np.zeros(numArms)
            arms_success = np.ones(numArms)
            arms_failure = np.ones(numArms)

            for i in range(horizon):
                # Getting rewards for given arm from environment
                arm = ts(numArms, arms_success, arms_failure)

                # Updating means of arms
                success = get_rewards(arm_true_means, arm)
                arm_rewards[arm] += success
                arms_success[arm] += success
                arms_failure[arm] += 1 - success

            regret.append(max_reward - np.sum(arm_rewards))

    else:
        regret = []

        for n in range(nRuns):
            arm_rewards = np.zeros(numArms)
            arms_success = np.ones(numArms)
            arms_failure = np.ones(numArms)

            for i in range(horizon):
                # Getting rewards for given arm from environment
                arm = ts(numArms, arms_success, arms_failure)

                # Updating means of arms
                success = get_rewards(arm_true_means, arm)
                arm_rewards[arm] += success
                arms_success[arm] += success
                arms_failure[arm] += 1 - success

            regret.append(max_reward - np.sum(arm_rewards))

    print regret
    regret_horizon.append(np.average(regret))

print regret_horizon
