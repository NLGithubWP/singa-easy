
import matplotlib.pyplot as plt
import numpy as np
import numpy


def drawBasic(plist, tlist, x, slicerate):

     y = []
     label = []
     pots = []

     for k in range(len(plist)):
          interval0 = [plist[k] if (ele/8 < 1/tlist[k]) else 0 for ele in x]
          interval1 = [plist[k] if (ele/8 > 1/tlist[k]) else 0 for ele in x]
          y1 = interval0 + ((1/tlist[k]) / (x/8)) * interval1
          y.append(y1)
          label.append('Sub-model with $r_i$ = ' + str(slicerate[k]) + ", $p_i$ = " + str(100*plist[k])[:5])

          tmp = []
          for x_ in x[10:]:
               if (x_/8) < 1/tlist[k]:
                    y_= plist[k]
               else:
                    y_ = plist[k]*((1/tlist[k]) / (x_/8))
               tmp.append([x_, y_])
          pots.append(tmp)

     return y, label, pots


x = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 150, 160, 170, 180, 200, 250, 300, 350, 400, 450, 550, 600, 650, 700, 750, 800, 1000]
x = [ele*32 for ele in x]
x = numpy.array(x)

y, label, pots = drawBasic([0.7937, 0.7188, 0.7094, 0.6512], [0.00141, 0.00110, 0.00071, 0.00049], x, [1, 0.75, 0.5, 0.25])

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)
ax.axis([0, 1000*32, 0.1, 0.9])

line_sym = ["-", "--", "-.", "-"]
pot_sym = ["o", ">", "<", "x"]
colors = ['blue', 'brown', 'aqua', 'green']
for i in range(len(y)):
     ax.plot(x, y[i], line_sym[i], marker=pot_sym[i], ms=7, label=label[i], linewidth=2, color=colors[i])

for k in range(len(pots)):
     ele = pots[k]
     for i in range(len(ele)):
          xc1, yc1 = ele[i]
          ax.plot(xc1, yc1, line_sym[k], marker=pot_sym[k], ms=10, color=colors[k])


ySch = [0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7937, 0.7911534375, 0.774429546875, 0.744321275, 0.7242490937500001, 0.7099126607142857, 0.6869111562500001, 0.6869111562500001, 0.6040653068181817, 0.5537265312499999, 0.5111321826923078, 0.4746227410714286, 0.44298122500000003, 0.4152948984375, 0.33223591874999997]
ax.plot(x, ySch, "--", marker='*', ms=12, label="Combination of sub-models", linewidth=2)


plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel('$W_{exp}=N/D$ (# instances / deadline constraint)', fontsize=20)
ax.set_ylabel('Effective accuracy %', fontsize=20)
plt.grid(linestyle='-.')
plt.savefig('./effective accuracy.jpg')

plt.show()









