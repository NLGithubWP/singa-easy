
import matplotlib.pyplot as plt
import numpy


def drawBasic(tlist, x):

     y = []
     label = []
     pots = []

     for k in range(len(tlist)):
          interval0 = [i*500 if (i < (1/tlist[k])/500) else 0 for i in x]
          interval1 = [1/tlist[k] if (i > (1/tlist[k])/500) else 0 for i in x]
          y1 = numpy.array(interval0) + numpy.array(interval1)
          y.append(y1)
          label.append('sub-model with process time(millisecond) ' + str(tlist[k]))
          for x_ in x:
               if x_ <= (1 / tlist[k]) / 500:
                    y_ = x_ * 500
               else:
                    y_ = 1 / tlist[k]
               pots.append([x_, y_])

     return y, label, pots


x = numpy.linspace(0, 10, 10)
print(x)
y, label, pots = drawBasic([0.0126, 0.0071, 0.00315, 0.00084], x)
print(y)

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)
ax.axis([0, 10, 0, 1300])


for i in range(len(y)):
     ax.plot(x, y[i], label=label[i], linewidth=2)

ax.plot(x, y[len(y)-1], "--" ,marker='*', ms=10, label="combine of sub-model with scheduling", linewidth=2)


for j in range(len(pots)):
     xc1, yc1 = pots[j]
     ax.scatter(xc1, yc1, s=50, facecolors='none', edgecolors='r')


plt.legend(fontsize=12)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel('Number of Delayed Data', fontsize=20)
ax.set_ylabel('RTT', fontsize=20)
plt.savefig('./throughSpeed.jpg')
plt.show()
# 每妙发送500条








