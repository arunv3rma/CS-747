import numpy as np
from pulp import *
from math import floor, ceil
import argparse

# ########## Reading Cmd line Arguments ##########

# python policyIteration.py --randomseed 89 --batchsize 8 --mdp /home/dataX/work/courses/CS-747/assignment2/data/MDP50.txt --algorithm rpi
parser = argparse.ArgumentParser()
parser.add_argument("--mdp")
parser.add_argument("--algorithm")
parser.add_argument("--batchsize")
parser.add_argument("--randomseed")
args = parser.parse_args()


# ########### Reading MDP file ###################
if not args.mdp:
    print "Please give correct path to MDP file with --mdp argument"
    exit()
mdp = open(args.mdp)

states = int(mdp.readline())
actions = int(mdp.readline())
# print states, actions

rewards = np.zeros((states, actions, states))
for s in range(states):
    for a in range(actions):
        reward = map(float, mdp.readline().split())
        for s_ in range(states):
            rewards[s, a, s_] = reward[s_]
# print rewards

transit_prob = np.zeros((states, actions, states))
for s in range(states):
    for a in range(actions):
        states_prob = map(float, mdp.readline().split())
        for s_ in range(states):
            transit_prob[s, a, s_] = states_prob[s_]
# print transit_prob

discounting_factor = float(mdp.readline())
# print discounting_factor


# ###################### Starting LP Code #########################


def lp_solve(reward, prob):
    min_states_value = pulp.LpProblem("Minimize value function, V(s)", pulp.LpMinimize)
    states_value = pulp.LpVariable.dicts('V(s)', range(states), cat='Continuous')
    min_states_value += (pulp.lpSum(states_value[i] for i in range(states)))

    for s in range(states):
        for a in range(actions):
            min_states_value += states_value[s] >= pulp.lpSum(
                prob[s, a, s_] * (reward[s, a, s_] + discounting_factor * states_value[s_]) for s_ in range(states))

    # Solving MDP
    min_states_value.solve()
    # print("Solution Type: ", pulp.LpStatus[min_states_value.status])

    policy_state_value = np.zeros(states)
    for state_value in min_states_value.variables():
        policy_state_value[int(state_value.name.split("_")[1])] = state_value.varValue

    # print policy_state_value

    states_actions_value = []
    policy = []
    for s in range(states):
        state_actions_value = []
        for a in range(actions):
            state_action_value = 0
            for s_ in range(states):
                state_action_value += prob[s, a, s_] * (rewards[s, a, s_] + discounting_factor * policy_state_value[s_])
            state_actions_value.append(state_action_value)

        states_actions_value.append(state_actions_value)
        policy.append(state_actions_value.index(max(state_actions_value)))

    # print(states_actions_value)

    # returning 1 to symmetric with other algorithms
    return policy_state_value, policy, 1


# ############ Functions needed for Policy Iteration ##############


def compute_policy_value(reward, prob, policy_action):
    coefficient_matrix = np.zeros((states, states))
    constant_terms = np.zeros(states)

    for s in range(states):
        coefficient_matrix[s, s] += 1
        for s_ in range(states):
            constant_terms[s] += reward[s, int(policy_action[s]), s_] * prob[s, int(policy_action[s]), s_]
            coefficient_matrix[s, s_] -= discounting_factor * prob[s, int(policy_action[s]), s_]

    return np.linalg.solve(coefficient_matrix, constant_terms)


def compute_action_value(reward, prob, policy_value):
    action_values = np.zeros((states, actions))
    for s in range(states):
        for a in range(actions):
            for s_ in range(states):
                action_values[s, a] += reward[s, a, s_] * prob[s, a, s_] + \
                                       discounting_factor * prob[s, a, s_] * policy_value[s_]
    return action_values


def find_improvable_states(reward, prob, policy_action):
    improved_states = np.zeros(states, int)
    policy_value = compute_policy_value(reward, prob, policy_action)
    action_values = compute_action_value(reward, prob, policy_value)

    for s in range(states):
        for a in range(actions):
            if action_values[s, a] - policy_value[s] > 1e-8:
                improved_states[s] = 1
    return improved_states


# ##################### Howard's PI ###############################


def howard_pi(reward, prob):
    policy = np.zeros(states)
    iter_count = 1
    while True:
        improvable_states = find_improvable_states(reward, prob, policy)
        if 1 not in improvable_states:
            break
        iter_count += 1
        for s in range(states):
            if improvable_states[s] == 1:
                policy[s] = 1 - policy[s]

    best_policy_values = compute_policy_value(rewards, transit_prob, policy)
    return best_policy_values, policy, iter_count


# ############## Mansour and Singh's Randomised PI ##################


def randomised_pi(reward, prob):
    if not args.randomseed:
        print "Please give random seed value with --randomseed argument"
        exit()
    np.random.seed(int(args.randomseed))

    policy = np.zeros(states)
    iter_count = 1
    while True:
        improvable_states = find_improvable_states(reward, prob, policy)
        # print improvable_states
        if 1 not in improvable_states:
            break

        iter_count += 1

        # Finding random subsets of states for changing
        indices = [i for i, x in enumerate(improvable_states) if x == 1]
        # print indices

        # subset = np.random.choice(sum(improvable_states), randint(1, sum(improvable_states)),
        #                           replace=False) int(floor(np.random.uniform(0, sum(l)) + 1))
        subset = np.random.choice(sum(improvable_states), int(floor(np.random.uniform(0, sum(improvable_states)) + 1)),
                                  replace=False)
        # print subset

        for i in subset:
            policy[indices[i]] = 1 - policy[indices[i]]

    best_policy_values = compute_policy_value(rewards, transit_prob, policy)
    return best_policy_values, policy, iter_count


# ########### Mansour and Singh's Batch-switching PI ################


def batch_switching_pi(reward, prob):
    if not args.batchsize:
        print "Please give batch size value with --batchsize argument"
        exit()
    batch_size = int(args.batchsize)
    if batch_size > states:
        print "Batch size value can not larger than number of states"
        exit()

    policy = np.zeros(states)
    iter_count = 1
    while True:
        improvable_states = find_improvable_states(reward, prob, policy)
        if 1 not in improvable_states:
            break
        iter_count += 1

        # Checking batches for improvable state from right to left
        start_index = len(improvable_states) - 1
        if len(improvable_states) % batch_size == 0:
            end_index = start_index - batch_size
        else:
            end_index = start_index - len(improvable_states) % batch_size
        # print start_index, end_index

        improvable_batched_states = np.zeros(states)
        batch_found = False
        for batch in range(int(ceil(1.0 * len(improvable_states) / batch_size)), 0, -1):
            for s in range(start_index, end_index, -1):
                if improvable_states[s] == 1:
                    improvable_batched_states[s] = improvable_states[s]
                    batch_found = True

            if batch_found:
                break
            start_index = end_index
            end_index -= batch_size

        # print improvable_states
        for s in range(states):
            if improvable_batched_states[s] == 1:
                policy[s] = 1 - policy[s]

    best_policy_values = compute_policy_value(rewards, transit_prob, policy)
    return best_policy_values, policy, iter_count


# ####################### Solving MDP and Get results ########################
if not args.algorithm:
    print "Please choose algorithm (lp, hpi, rpi, bspi) with --algorithm argument"
    exit()

if args.algorithm == "lp":
    values, actions, iterations = lp_solve(rewards, transit_prob)

elif args.algorithm == "hpi":
    values, actions, iterations = howard_pi(rewards, transit_prob)

elif args.algorithm == "rpi":
    values, actions, iterations = randomised_pi(rewards, transit_prob)

elif args.algorithm == "bspi":
    values, actions, iterations = batch_switching_pi(rewards, transit_prob)

else:
    print "Please choose correct algorithm from (lp, hpi, rpi, bspi)"
    exit()

# print "Number of iterations =", iterations
for s in range(states):
    print str(values[s]) + "\t" + str(int(actions[s]))
