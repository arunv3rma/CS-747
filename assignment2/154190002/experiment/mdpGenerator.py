import numpy as np


states = 50
actions = 2
mdp_instances = 100

for i in range(1, mdp_instances+1):
    np.random.seed(i)
    fileName = "input/mdp" + str(i) +".txt"
    mdpStore = open(fileName, 'w')

    mdpStore.write(str(states) + "\n")
    mdpStore.write(str(actions) + "\n")

    for s in range(0, states):
        for a in range(0, actions):
            for s_ in range(0, states):
                mdpStore.write(str(round(np.random.uniform(-1, 1), 6)) + "\t",)

            mdpStore.write("\n",)

    for s in range(0, states):
        for a in range(0, actions):
            state_prob = 0
            prob = np.random.uniform(0, 1, states)
            for s_ in range(0, states-1):
                state_state_prob = round(prob[s_]/sum(prob), 6)
                state_prob += state_state_prob
                mdpStore.write(str(state_state_prob) + "\t",)

            mdpStore.write(str(1 - state_prob) + "\t",)
            mdpStore.write("\n",)

    mdpStore.write(str(round(np.random.uniform(0, 1), 2)))