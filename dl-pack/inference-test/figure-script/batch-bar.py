import matplotlib.pyplot as plt
import numpy as np
from pylab import *

# Input data; groupwise
resnet = [112.17333780279539, 76.96422913329054, 68.83871653439321]
mobilenet = [55.760545663535595,  45.636449311006196,  47.180608652842544]
pack = [137.58277589583304, 100.14077688253019, 89.242348792313]

labels = ['10', '100', '1000']
ylabels = np.arange(0, 181, 20)
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
plt.ylim([0,180])
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
plt.savefig('batch-bar.png',dpi=300)
