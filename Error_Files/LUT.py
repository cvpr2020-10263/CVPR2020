from __future__ import print_function
import numpy as np
import random

LUT = []
zeros = []
for i in range(0,30000000):
    a = 0
    for x in xrange(0,31):
        rand = random.uniform(0, 1)

        if rand <= 0.06:
            a = a + 1 * (2 ** x)


        else:
            a = a + 0

    LUT.append(a)
    if a == 0:
        zeros.append(0)



print (len(LUT))
print (len(zeros))

LUT = np.array((LUT))
np.save("LUT1_0.06", LUT)

