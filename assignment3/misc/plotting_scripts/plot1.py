import numpy as np
import matplotlib.pyplot as plt

# Instance 0, QLearning
Q_1 = []
for r in range(10):
    fileName = "results/" + "qlearning1_rs" + str(r) + ".txt"
    resultFile = open(fileName)
    print resultFile.readline()
    rewards = map(int, list(resultFile.readline().split("[")[1].split("]")[0].split(", ")))
    resultFile.close()
    Q_1.append(rewards)


S_1 = []
for r in range(10):
    fileName = "results/" + "sarsa1_accum_rs" + str(r) + ".txt"
    resultFile = open(fileName)
    print resultFile.readline()
    rewards = map(int, list(resultFile.readline().split("[")[1].split("]")[0].split(", ")))
    resultFile.close()
    S_1.append(rewards)

X = range(1, 1501)
Q = np.mean(np.array(Q_1), axis=0)
S = np.mean(np.array(S_1), axis=0)


plt.plot(X, S, 'r--', label="Sarsa(lambda)")
plt.plot(X, Q, 'g--', label='Q-Learning')

plt.legend(loc='lower right', numpoints=1)
plt.legend(loc='lower right', numpoints=1)
# loc is to specify the legend location
plt.title("Average Rewards vs Episode number Plot for Instance 1")
plt.ylabel("Average Rewards of each Episode")
plt.xlabel("Episode Number")

plt.savefig("qs1t.png", bbox_inches='tight')
# plt.show()
plt.close()