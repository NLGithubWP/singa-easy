import random

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np



def drawBasic(tlist, x, slicerate):

     y = []
     label = []
     pots = []

     for k in range(len(tlist)):
          interval0 = [x_i *tlist[k]  for x_i in x]
          y_1 = interval0
          y.append(y_1)
          label.append('sub-model with $r_i$ = ' + str(slicerate[k]) + ", $t_i=$" + str(tlist[k]) + "$s$")
          if k==0:
               tmp = []
               for x_ in x[k:]:
                    if x_ < 1/tlist[k]:
                         y_= x_
                    else:
                         y_ = 1/tlist[k]
                    tmp.append([x_, y_])
               pots.append(tmp)
          if k==1:
               tmp = []
               for x_ in x[k:]:
                    if x_ < 1/tlist[k]:
                         y_= x_
                    else:
                         y_ = 1/tlist[k]
                    tmp.append([x_, y_])
               pots.append(tmp)
          if k>1:
               tmp = []
               for x_ in x[k:]:
                    if x_ < 1/tlist[k]:
                         y_= x_
                    else:
                         y_ = 1/tlist[k]
                    tmp.append([x_, y_])
               pots.append(tmp)

     return y, label, pots

import numpy

x = [1, 50, 80, 100, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1100, 1200, 1300, 1600, 1700, 1900, 2100, 2300, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3900, 4000]
x = numpy.array(x)
print(x)
y, label, pots = drawBasic([0.0126, 0.0071, 0.00315, 0.00079], x, [1, 0.75, 0.5, 0.25])


fig=plt.figure(figsize=(10, 10))
ax=fig.add_subplot(111)
ax.axis([0, 1600, 0, 2])

line_sym = ["-", "--", "-.", "-"]
pot_sym = ["o", ">", "<", "x"]
colors = ['blue', 'brown', 'aqua', 'green']
for i in range(len(y)):
     y_i = [ele-0.01 for ele in y[i]]
     ax.plot(x, y_i, line_sym[i], marker=pot_sym[i], ms=10, label=label[i], linewidth=2,  color=colors[i])


ysch_ = [0.0126, 0.63, 0.997, 0.996, 0.996, 0.99735, 0.9963500000000001, 0.99804, 0.99916, 0.99792, 0.99904, 0.9978, 0.99892, 0.9988, 0.99992, 1.027, 1.264, 1.343, 1.501, 1.6589999999999998, 1.817, 1.975, 2.0540000000000003, 2.133, 2.2119999999999997, 2.291, 2.37, 2.449, 2.528, 2.607, 2.686, 3.0810000000000004, 3.16]

ysh = [ele-0.01 if (ele<1200) else ele for ele in ysch_]

ax.plot(x, ysh,"--", marker='*', ms=10, label="combine of sub-models", linewidth=2)

plt.hlines(1.01, 0, 4500, label="deadline constrain=1s", linewidth=4, color="red")
#
plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel('Ingesting rate(#examples/second)', fontsize=20)
ax.set_ylabel('Latency(second)', fontsize=20)
plt.grid(linestyle='-.')
plt.savefig('./latency.jpg')
plt.show()








