import matplotlib.pyplot as plt
import numpy as np

y = [399.6, 250, 124, 113.884615385, 84.956521739]
x = [1,  5,  10,  15,  20]

plt.rc('xtick', labelsize=20)     
plt.rc('ytick', labelsize=20)

plt.figure(figsize=(5, 3))
plt.plot(x,y,marker="^")
plt.ylabel('waiting time (unit time)',)
plt.xlabel('time rate')
plt.xlim(0, 21)

plt.xticks([1,5,10,15,20])
plt.yticks([0,100,200,300,400])

ax = plt.gca()
ax.tick_params(axis='both',which='major',direction='in', top=True, right = True)
ax.grid(linestyle='--', linewidth='0.5')
ax.set_xticklabels([1,5,10,15,20], fontsize=14)
ax.set_yticklabels([0,100,200,300,400], fontsize=14)
ax.set_ylabel("Waiting time (unit time)", fontsize=14)
ax.set_xlabel("Time interval between batches", fontsize=14)
plt.tight_layout()
#plt.show()
plt.savefig('wt.pdf')




