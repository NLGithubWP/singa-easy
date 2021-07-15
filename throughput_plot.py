
import matplotlib.pyplot as plt


def drawBasic(tlist, x):

     y = []
     label = []
     pots = []

     for k in range(len(tlist)):
          interval0 = [i if (i < 1/tlist[k]) else 0 for i in x]
          interval1 = [1 if (i > 1/tlist[k]) else 0 for i in x]
          y1 = numpy.array(interval0) + (1/tlist[k]) * numpy.array(interval1)
          y.append(y1)
          label.append('sub-model with process time(millisecond) ' + str(tlist[k]))
          if k==0:
               for x_ in x[k:]:
                    if x_ < 1/tlist[k]:
                         y_= x_
                    else:
                         y_ = 1/tlist[k]
                    pots.append([x_, y_])
          if k==1:
               for x_ in x[k:]:
                    if x_ < 1/tlist[k]:
                         y_= x_
                    else:
                         y_ = 1/tlist[k]
                    pots.append([x_, y_])
          if k>1:
               for x_ in x[k:]:
                    if x_ < 1/tlist[k]:
                         y_= x_
                    else:
                         y_ = 1/tlist[k]
                    pots.append([x_, y_])

     return y, label, pots


import numpy
x = numpy.array([0, 1, 50, 80, 100, 200, 300, 400, 540, 800,  910, 1020, 1282, 1500])
print(x)
y, label, pots = drawBasic([0.0126, 0.0071, 0.00315, 0.00084], x)


fig=plt.figure(figsize=(10, 10))
ax=fig.add_subplot(111)
ax.axis([0, 1550, 0, 1400])

for i in range(len(y)):
     ax.plot(x, y[i], label=label[i], linewidth=2)

ax.plot(x, y[len(y)-1], label="Combine of model", linewidth=2)



for j in range(len(pots)):
     xc1, yc1 = pots[j]
     ax.scatter(xc1, yc1, s=50, facecolors='none', edgecolors='r')


plt.legend(fontsize=13)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel('Send Rate(data/second)', fontsize=20)
ax.set_ylabel('Throughput(data/second)', fontsize=20)
plt.savefig('./throughput.jpg')
plt.show()








