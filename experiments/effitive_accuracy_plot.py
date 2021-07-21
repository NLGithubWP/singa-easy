
import matplotlib.pyplot as plt
import numpy as np


def drawBasic(plist, tlist, x, slicerate):

     y = []
     label = []
     pots = []

     for k in range(len(plist)):
          interval0 = [plist[k] if (i < 1/tlist[k]) else 0 for i in x]
          interval1 = [plist[k] if (i > 1/tlist[k]) else 0 for i in x]
          y1 = interval0 + ((1/tlist[k]) / x) * interval1
          y.append(y1)
          # label.append('sub-model with accuracy ' + str(100*plist[k])[:5])
          label.append('sub-model with $r_i$ = ' + str(slicerate[k]) + ", $p_i$ = " + str(100*plist[k])[:5])
          if k==0:
               tmp = []
               for x_ in x[k:]:
                    if x_ < 1/tlist[k]:
                         y_= plist[k]
                    else:
                         y_ = plist[k]*((1/tlist[k]) / x_)
                    tmp.append([x_, y_])
               pots.append(tmp)
          if k==1:
               tmp = []
               for x_ in x[k+3:]:
                    if x_ < 1/tlist[k]:
                         y_= plist[k]
                    else:
                         y_ = plist[k]*((1/tlist[k]) / x_)
                    tmp.append([x_, y_])
               pots.append(tmp)
          if k>1:
               tmp = []
               for x_ in x[k:]:
                    if x_ < 1/tlist[k]:
                         y_= plist[k]
                    else:
                         y_ = plist[k]*((1/tlist[k]) / x_)
                    tmp.append([x_, y_])
               pots.append(tmp)

     return y, label, pots


def drawSchedulingOne(plist, tlist):
     pass


import numpy
x = numpy.array([0, 1, 50, 80, 100, 200, 300, 400, 540, 800,  910, 1020, 1282, 1500])
print(x)
y, label, pots = drawBasic([0.7609, 0.7374, 0.7109, 0.6391], [0.0126, 0.0071, 0.00315, 0.00084], x, [1, 0.75, 0.5, 0.25])


fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)
ax.axis([0, 1550, 0, 1])
# plt.xlim((0, 1500))
# plt.ylim((0, 1))
# plt.xlabel('qs=(number of query data) / (deadline constraint)')
# plt.ylabel('effective accuracy')

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


ySch = [0.7609, 0.7609, 0.7609, 0.7603125000000001, 0.7496200000000001, 0.7232225, 0.7120483333333334, 0.6906165, 0.6704792592592592, 0.6517547499999999, 0.6470690109890109, 0.6433939215686274, 0.5932363494539781, 0.5070193333333333]
ax.plot(x, ySch, "--", marker='*', ms=12, label="combination of sub-models", linewidth=2)

# for j in range(len(ySch)):
#      ax.scatter(x[j], ySch[j], s=50, facecolors='none', edgecolors='r')


plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel('$W_{exp}=N/D$ (# examples / deadline constraint)', fontsize=20)
ax.set_ylabel('effective accuracy %', fontsize=20)
plt.grid(linestyle='-.')
plt.savefig('./effective accuracy.jpg')

plt.show()









