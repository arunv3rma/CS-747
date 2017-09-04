from pulp import *
import numpy as np
train_features = []
train_labels = []
X = np.array(train_features, dtype = float)
Y = np.array(train_labels, dtype = float)

M = range(len(X[:, :]))
P = range(8)


Prob = LpProblem("HalfSpace", LpMinimize)


# Defining the variables
# w = LpVariable.dicts("w", P, -15, 10)
w = LpVariable.dicts("w", P)
z = LpVariable.dicts("z", M,0)
v = LpVariable.dicts("v", M,0)


# The Objective Function
# Prob += sum([(w[j]*X[i][j]*Y[j]) for i in M for j in P])
# for i in M : Prob += sum([w[j]*X[i][j]*Y[j] for j in P]) >= 0

Prob += sum([z[i] for i in M])
for i in M : Prob += sum([w[j]*X[i][j]*Y[i] for j in P]) >= 1 - z[i]

# Prob += sum([z[i] + v[i] for i in M])
# for i in M : Prob += sum([w[j]*X[i][j]*Y[j] for j in P]) >= 1- z[i] + v[i]


Prob.solve()
# print sum([(z[i].varValue + v[i].varValue) for i in M])
print sum([z[i].varValue for i in M])
print("Total error= ", value(Prob.objective))

weights = []
for j in P:
    weights.append(w[j].varValue)

print weights
k=0

print "Training Error"
correct_predict = 0
for i in range(615):
    wX = 0
    for j in range(len(P)):
        wX += train_features.iat[i,j] * weights[j]

    if (train_labels[i] * wX) >= 1:
        correct_predict += 1

print "Training Accuracy: ", (correct_predict/615.0) * 100, "%"

print "Test error"
correct_predict = 0
for i in range(153):
    wX = 0
    for j in range(len(P)):
        wX += test_features.iat[i,j] * weights[j]

    if (test_labels[i] * wX) >= 1:
        correct_predict += 1
#
print "Test Accuracy: ", (correct_predict/153.0) * 100, "%"