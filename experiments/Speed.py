


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np


def drawBasic(tlist, x, slicerate):

     y = []
     label = []
     pots = []

     for k in range(len(tlist)):
          interval0 = [i if (i < 1/tlist[k]) else 0 for i in x]
          interval1 = [1 if (i > 1/tlist[k]) else 0 for i in x]
          y1 = numpy.array(interval0) + (1/tlist[k]) * numpy.array(interval1)
          y.append(y1)
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
x = numpy.array([0, 1, 50, 80, 100, 200, 300, 400, 540, 800,  910, 1020, 1282, 1500])
print(x)
y, label, pots = drawBasic([0.0126, 0.0071, 0.00315, 0.00084], x, [1, 0.75, 0.5, 0.25])


fig=plt.figure(figsize=(10, 10))
ax=fig.add_subplot(111)
ax.axis([0, 1550, 0, 1400])

line_sym = ["-", "--", "-.", "-"]
pot_sym = ["o", ">", "<", "x"]
colors = ['blue', 'brown', 'aqua', 'green']
for i in range(len(y)):
     ax.plot(x, y[i], line_sym[i], marker=pot_sym[i], ms=6, label=label[i], linewidth=2,  color=colors[i])

ax.plot(x, y[len(y)-1], label="combination of sub-models", linewidth=2)



for k in range(len(pots)):
     ele = pots[k]
     for i in range(len(ele)):
          xc1, yc1 = ele[i]
          ax.plot(xc1, yc1, line_sym[k], marker=pot_sym[k], ms=10, color=colors[k])


# for j in range(len(pots)):
#      xc1, yc1 = pots[j]
#      ax.scatter(xc1, yc1, s=50, facecolors='none', edgecolors='r')


plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel('Ingesting rate(#examples/second)', fontsize=20)
ax.set_ylabel('Single model maximum capacity', fontsize=20)
plt.grid(linestyle='-.')
plt.savefig('./capacity.jpg')
plt.show()







