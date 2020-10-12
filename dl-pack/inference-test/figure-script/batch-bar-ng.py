import matplotlib.pyplot as plt
import numpy as np
from pylab import *

# Input data; groupwise
resnet = [118.3792133028619, 74.47076625512757, 67.22564806090668]
mobilenet = [80.35949564678594, 48.1942391556222,  46.812860125636995]
pack = [150.12085809430573, 102.83475236854672, 89.242348792313]

labels = ['10', '100', '1000']
ylabels = np.arange(0, 161, 20)
# Setting the positions and width for the bars
pos = list(range(len(resnet))) 
width = 0.25 # the width of a bar
    
# Plotting the bars
fig, ax = plt.subplots(figsize=(7,4.2))

bar1=plt.bar(pos, resnet, width, edgecolor='k',
                 alpha=0.5,
                 color='#b04e0f',
                 hatch='//', # this one defines the fill pattern
                 label=labels[0])

plt.bar([p + width for p in pos], mobilenet, width,edgecolor='k',
                 alpha=0.5,
                 color='#5d06e9',
                 hatch='x',
                 label=labels[1])
    
plt.bar([p + width*2 for p in pos], pack, width,edgecolor='k',
                 alpha=0.5,
                 color='#045c5a',
                 hatch='.',
                 label=labels[2])



ax.set_xticks([p + 1 * width for p in pos])
ax.set_xticklabels(labels, fontsize=14)
#plt.ylabel('Fraction of correctly complete the schema', )
#plt.xlabel('Number of input web tables', )

# Setting the x-axis and y-axis limits
plt.xlim([-0.2,2.7])
plt.ylim([0,160])
#plt.yticks()
ax.set_yticklabels(ylabels, fontsize=14)

# Setting axis labels and ticks
ax.set_ylabel('Latency',fontsize=16)
ax.set_xlabel('Batch size',fontsize=16)

plt.tick_params(axis='x',bottom='off')

plt.tick_params(axis='y',direction='in')

# Adding the legend and showing the plot
plt.legend(['ResNet', 'MobileNet', 'Packed'], loc='upper right')
plt.tight_layout()
#plt.show()
plt.savefig('batch-bar-ng.png',dpi=300)
