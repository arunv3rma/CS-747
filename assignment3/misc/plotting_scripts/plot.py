import numpy as np
import matplotlib.pyplot as plt

# Instance 0, QLearning
Q_0 = []
for r in range(10):
    fileName = "results/" + "qlearning_rs" + str(r) + ".txt"
    resultFile = open(fileName)
    print resultFile.readline()
    rewards = map(int, list(resultFile.readline().split("[")[1].split("]")[0].split(", ")))
    resultFile.close()
    Q_0.append(rewards)

# Instance 0, QLearning
Q_1 = []
for r in range(10):
    fileName = "results/" + "qlearning1_rs" + str(r) + ".txt"
    resultFile = open(fileName)
    print resultFile.readline()
    rewards = map(int, list(resultFile.readline().split("[")[1].split("]")[0].split(", ")))
    resultFile.close()
    Q_1.append(rewards)

X = range(1, 1501)
Y = np.mean(np.array(Q_1), axis=0)
# plt.plot(X, Y, 'b--v')
plt.plot(X, Y)


# loc is to specify the legend location
plt.title("No. of Iterations vs Batch Size in Mansour and Singh's Batch-switching PI")
plt.ylabel("Number of Iterations")
plt.xlabel("Batch Size")

plt.savefig("Iteration_Batch1.png", bbox_inches='tight')
# plt.show()
plt.close()