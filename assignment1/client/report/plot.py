import matplotlib.pyplot as plt

horizons = [10, 100, 1000, 10000, 100000]

# Plotting epsilon-greedy Performance for 5 Instances
e_005 = [4.0099999999999998, 40.140000000000001, 389.47000000000003, 2617.1799999999998, 7501.9399999999996]
e_01 = [4.0499999999999998, 39.68, 367.61000000000001, 1958.75, 5055.9300000000003]
e_05 = [3.9500000000000002, 38.43, 280.88, 717.39999999999998, 1570.01]
e_1 = [3.8599999999999999, 37.210000000000001, 190.36000000000001, 487.32999999999998, 898.46000000000004]
e_5 = [3.8100000000000001, 25.75, 70.439999999999998, 247.06999999999999, 1093.6900000000001]
e_10 = [3.25, 15.800000000000001, 53.950000000000003, 334.19999999999999, 3021.1399999999999]
e_15= [3.3100000000000001, 15.470000000000001, 54.619999999999997, 344.75, 3029.0500000000002]
e_20 = [3.2999999999999998, 15.93, 55.630000000000003, 333.47000000000003, 3041.8800000000001]
e_30 = [3.5699999999999998, 15.890000000000001, 62.170000000000002, 347.86000000000001, 3037.4400000000001]
e_50 = [3.1800000000000002, 16.34, 62.049999999999997, 331.81, 3039.73]
e_75 = [3.3199999999999998, 17.629999999999999, 60.759999999999998, 330.91000000000003, 3049.4400000000001]
e_90 = [3.3500000000000001, 15.51, 66.120000000000005, 339.06999999999999, 3051.0599999999999]

plt.plot(horizons, e_005,'b--H', label='e=0.0005')
plt.plot(horizons, e_01,'c-D', label='e=0.001')
plt.plot(horizons, e_05,'r--d', label='e=0.005')
plt.plot(horizons, e_1,'y--X', label='e=0.01')
plt.plot(horizons, e_5,'g--^', label='e=0.05')
plt.plot(horizons, e_10,'b--v', label='e=0.10')
plt.plot(horizons, e_15,'c--+', label='e=0.15')
plt.plot(horizons, e_20,'r--*', label='e=0.20')
plt.plot(horizons, e_30,'y--<', label='e=0.30')
plt.plot(horizons, e_50,'k-->', label='e=0.50')
plt.plot(horizons, e_75,'m--s', label='e=0.75')
plt.plot(horizons, e_90,'g--h', label='e=0.90')

# loc is to specify the legend location
plt.legend(loc='upper left', numpoints=1)
plt.title("epsilon-greedy algorithm for different epsilon(e) values for 5 Arms")
plt.ylabel("Expected Cumulative Regret")
plt.xlabel("Horizon")

plt.savefig("Epsilon_Greedy_5.png", bbox_inches='tight')
# plt.show()
plt.close()

# Plotting epsilon-greedy Performance for 25 Instances
e_005_25 = [8.4300000000000139, 84.370000000000005, 802.19000000000005, 4650.3400000000001, 16907.529999999999]
e_01_25 = [8.4800000000000129, 84.629999999999995, 752.92999999999995, 3644.3600000000001, 12545.139999999999]
e_05_25 = [8.3700000000000117, 78.489999999999995, 460.41000000000003, 1555.4000000000001, 5113.6800000000003]
e_1_25 = [8.0700000000000109, 74.159999999999997, 373.77999999999997, 1032.1400000000001, 3741.4000000000001]
e_5_25 = [7.9800000000000102, 50.759999999999998, 156.34, 683.86000000000001, 3106.96]
e_10_25 = [6.8200000000000056, 32.210000000000001, 151.65000000000001, 913.72000000000003, 7806.1700000000001]
e_15_25= [6.6600000000000072, 33.0, 151.16, 897.63, 7781.3699999999999]
e_20_25 = [6.7400000000000055, 33.990000000000002, 159.19, 944.51999999999998, 7759.2799999999997]
e_30_25 = [6.7800000000000047, 34.219999999999999, 143.71000000000001, 901.59000000000003, 7831.9700000000003]
e_50_25 = [6.6600000000000046, 32.759999999999998, 158.78999999999999, 919.45000000000005, 7798.0299999999997]
e_75_25 = [6.9300000000000068, 30.5, 154.00999999999999, 908.91999999999996, 7796.8000000000002]
e_90_25 = [7.2100000000000053, 34.600000000000001, 153.68000000000001, 920.83000000000004, 7770.6400000000003]

plt.plot(horizons, e_005_25,'b--H', label='e=0.0005')
plt.plot(horizons, e_01_25,'c-D', label='e=0.001')
plt.plot(horizons, e_05_25,'r--d', label='e=0.005')
plt.plot(horizons, e_1_25,'y--X', label='e=0.01')
plt.plot(horizons, e_5_25,'g--^', label='e=0.05')
plt.plot(horizons, e_10_25,'b--v', label='e=0.10')
plt.plot(horizons, e_15_25,'c--+', label='e=0.15')
plt.plot(horizons, e_20_25,'r--*', label='e=0.20')
plt.plot(horizons, e_30_25,'y--<', label='e=0.30')
plt.plot(horizons, e_50_25,'k-->', label='e=0.50')
plt.plot(horizons, e_75_25,'m--s', label='e=0.75')
plt.plot(horizons, e_90_25,'g--h', label='e=0.90')

# loc is to specify the legend location
plt.legend(loc='upper left', numpoints=1)
plt.title("epsilon-greedy algorithm for different epsilon(e) values for 25 Arms")
plt.ylabel("Expected Cumulative Regret")
plt.xlabel("Horizon")

plt.savefig("Epsilon_Greedy_25.png", bbox_inches='tight')
# plt.show()
plt.close()

# Plotting comparison of Regret Bandit's algorithm for 5 Instance
u = [1.75, 13.33, 74.109999999999999, 186.53, 271.47000000000003]
t = [1.8899999999999999, 11.039999999999999, 28.899999999999999, 52.380000000000003, 49.060000000000002]
k = [2.3300000000000001, 12.6, 50.299999999999997, 125.45999999999999, 166.69999999999999]

plt.plot(horizons, e_01,'b--<', label='epsilon-greedy')
plt.plot(horizons, u,'r-->', label='UCB')
plt.plot(horizons, k,'g--s', label='KL-UCB')
plt.plot(horizons, t,'y--v', label='Thompson-Sampling')

# loc is to specify the legend location
plt.legend(loc='upper left', numpoints=1)
plt.title("Different bandits algorithm cumulative regret for 5 Arms")
plt.ylabel("Expected Cumulative Regret")
plt.xlabel("Horizon")

plt.savefig("Bandits_5.png", bbox_inches='tight')
# plt.show()
plt.close()



# Plotting comparison of Regret Bandit's algorithm for 25 Instance
u_25 = [7.3000000000000078, 38.659999999999997, 208.74000000000001, 642.25, 1196.03]
t_25 = [4.8999999999999924, 24.98, 50.539999999999999, 74.209999999999994, 102.88]
k_25 = [-1.8100000000000023, 23.600000000000001, 67.620000000000005, 146.0, 186.96]

plt.plot(horizons, e_05,'b--<', label='epsilon-greedy')
plt.plot(horizons, u_25,'r-->', label='UCB')
plt.plot(horizons, k_25,'g--s', label='KL-UCB')
plt.plot(horizons, t_25,'y--v', label='Thompson-Sampling')

# loc is to specify the legend location
plt.legend(loc='upper left', numpoints=1)
plt.title("Different bandits algorithm cumulative regret for 25 Arms")
plt.ylabel("Expected Cumulative Regret")
plt.xlabel("Horizon")

plt.savefig("Bandits_25.png", bbox_inches='tight')
# plt.show()
plt.close()
