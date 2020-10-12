import numpy as np
import time
#import random as rd


tr = 2

np.random.seed(int(time.time()))
a = np.random.randint(low=1,high=10,size=1000)
#np.random.seed(int(time.time()))
b = np.random.randint(low=1,high=10,size=1000)

#print(a)
#print(b)

a_sum = 0
b_sum = 0
count = 0
for i in range(1000):
    a_sum += a[i]
    b_sum += b[i]
    if a_sum != b_sum:
        a_sum += tr
        b_sum += tr
    else:
        count += 1

za = np.sum(a)
zb = np.sum(b)

wa = a_sum - za
wb = b_sum - zb

print(count)
print(wa)
print(wb)

