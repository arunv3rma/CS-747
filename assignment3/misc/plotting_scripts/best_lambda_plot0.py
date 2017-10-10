import numpy as np
import matplotlib.pyplot as plt

lambda_val = ['0', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.5', '0.6', '0.65', '0.7', '0.8', '0.85', '0.9', '1.0']
lambda_val = map(float, lambda_val)
lambda_val[0] = 0
print lambda_val
# Instance 0, Sarsa - accumulate
SR_0 = []
for lamb in lambda_val:
    lamb_reward = []
    for r in range(10):
        fileName = "results/" + "sarsa_replace_" + str(lamb) + "_rs" + str(r) + ".txt"
        resultFile = open(fileName)
        print resultFile.readline()
        rewards = map(int, list(resultFile.readline().split("[")[1].split("]")[0].split(", ")))
        resultFile.close()
        lamb_reward.append(1.0 * sum(rewards)/len(rewards))
    SR_0.append(1.0 * sum(lamb_reward)/len(lamb_reward))


SA_0 = []
for lamb in lambda_val:
    lamb_reward = []
    for r in range(10):
        fileName = "results/" + "sarsa_accum_" + str(lamb) + "_rs" + str(r) + ".txt"
        resultFile = open(fileName)
        print resultFile.readline()
        rewards = map(int, list(resultFile.readline().split("[")[1].split("]")[0].split(", ")))
        resultFile.close()
        lamb_reward.append(1.0 * sum(rewards)/len(rewards))
    SA_0.append(1.0 * sum(lamb_reward)/len(lamb_reward))


plt.plot(lambda_val, SA_0, 'g--h', label='Accumulating')
plt.plot(lambda_val, SR_0, 'r--v', label="Replacing")

plt.legend(loc='upper left', numpoints=1)
# loc is to specify the legend location
plt.title("Average Cumulative Rewards vs Lambda value Plot for Instance 0")
plt.ylabel("Average Cumulative Rewards of 500 Episodes")
plt.xlabel("Lambda Value")

plt.savefig("sarsa0.png", bbox_inches='tight')
# plt.show()
plt.close()