import numpy as np
import os
from math import ceil
import matplotlib.pyplot as plt

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


# ########### Mansour and Singh's Batch-switching PI ################


def batch_switching_pi(reward, prob, batch_size):
    if batch_size > states:
        # print "Batch size value can not larger than number of states"
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

        for s in range(states):
            if improvable_batched_states[s] == 1:
                policy[s] = 1 - policy[s]

    # best_policy_values = compute_policy_value(rewards, transit_prob, policy)
    # return best_policy_values, policy, iter_count
    return iter_count


# ####################### Comparing MDP PI Methods ########################

mdp_instances = 100

b_iterations_list = []
for i in range(1, mdp_instances+1):
    # ########### Reading MDP file ###################
    mdp_file = os.getcwd() + "/input/mdp" + str(i) + ".txt"
    # print "Solving ", mdp_file
    mdp = open(mdp_file)

    states = int(mdp.readline())
    actions = int(mdp.readline())

    rewards = np.zeros((states, actions, states))
    for s in range(states):
        for a in range(actions):
            reward = map(float, mdp.readline().split())
            for s_ in range(states):
                rewards[s, a, s_] = reward[s_]

    transit_prob = np.zeros((states, actions, states))
    for s in range(states):
        for a in range(actions):
            states_prob = map(float, mdp.readline().split())
            for s_ in range(states):
                transit_prob[s, a, s_] = states_prob[s_]

    discounting_factor = float(mdp.readline())

    batch_iteration = []
    for batch_size in range(1, states+1):
        batch_iteration.append(batch_switching_pi(rewards, transit_prob, batch_size))
    b_iterations_list.append(batch_iteration)

print b_iterations_list
avg_batch_iteration = np.mean(np.array(b_iterations_list), axis=0)
print avg_batch_iteration

X = range(1, 51)
Y = np.mean(np.array(b_iterations_list), axis=0)
plt.plot(X, Y, 'b--v')

# loc is to specify the legend location
plt.title("No. of Iterations vs Batch Size in Mansour and Singh's Batch-switching PI")
plt.ylabel("Number of Iterations")
plt.xlabel("Batch Size")

plt.savefig("Iteration_Batch.png", bbox_inches='tight')
# plt.show()
plt.close()