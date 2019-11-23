import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

x = list(range(100))
x = np.array(x)

y = list(range(100))
y = np.array(y)

T1 = 15
T2 = 100
T3 = 45
T4 = 60
T5 = 75
x1 = 0
#x1 = np.where(x > T5, (T5 + x) * 0.245, x)
#x1 = np.where((x >= T4) & (x <= T5), (T4 + x) * 0.273, x1)
#x1 = np.where((x >= T3) & (x <= T4), (T3 + x) * 0.312, x1)
#x1 = np.where((x >= T2) & (x <= T3), (T2 + x) * 0.375, x1)
x1 = np.where((x >= T1) & (x <= T2), (T1 + x) * 0.5, x)
with open('data.csv', mode='w') as datawriter:
    data = csv.writer(datawriter)
    for i in range(0, len(x1)):
        list1 = []
        list1.append(x1[i])
        list1.append(y[i])
        data.writerow(list1)

plt.plot(y, x1)
plt.show()

datawriter.close()
