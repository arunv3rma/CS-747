import numpy as np
import matplotlib.pyplot as plt

lambda_val = ['0.8', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96', '0.98', '1.0']
lambda_val = map(float, lambda_val)
SA_1 = []
for lamb in lambda_val:
    lamb_reward = []
    for r in range(10):
        if lamb == 0.9:
            fileName = "results/" + "sarsa1t_accum_" + str(lamb) + "0_rs" + str(r) + ".txt"
        else:
            fileName = "results/" + "sarsa1t_accum_" + str(lamb) + "_rs" + str(r) + ".txt"
        resultFile = open(fileName)
        print resultFile.readline()
        rewards = map(int, list(resultFile.readline().split("[")[1].split("]")[0].split(", ")))
        resultFile.close()
        lamb_reward.append(1.0 * sum(rewards)/len(rewards))
    SA_1.append(1.0 * sum(lamb_reward)/len(lamb_reward))


plt.plot(lambda_val, SA_1, 'g--h', label='Accumulating')

plt.legend(loc='upper left', numpoints=1)
# loc is to specify the legend location
plt.title("Average Cumulative Rewards vs Lambda value Plot for Instance 1")
plt.ylabel("Average Cumulative Rewards of 500 Episodes")
plt.xlabel("Lambda Value")

plt.savefig("sarsa1_t.png", bbox_inches='tight')
# plt.show()
plt.close()