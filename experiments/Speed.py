
import matplotlib.pyplot as plt
import numpy


def drawBasic(tlist, x, slicerate):

     y = []
     label = []
     pots = []

     for k in range(len(tlist)):
          interval0 = [1/tlist[k] if (r < (1/tlist[k])) else 0 for r in x]
          interval1 = [(1/tlist[k])*(1/tlist[k])/r if (r > (1/tlist[k])) else 0 for r in x]
          y1 = numpy.array(interval0) + numpy.array(interval1)
          y.append(y1)
          label.append('sub-model with $r_i$ = ' + str(slicerate[k]) + ", $t_i=$" + str(tlist[k]) + "$s$")


          tmp = []
          for x_ in x:
               if x_ < 1/tlist[k]:
                    y_ = 1/tlist[k]
               else:
                    y_ =(1/tlist[k])*(1/tlist[k])/x_
               tmp.append([x_, y_])
          pots.append(tmp)

     return y, label, pots


x = [1, 50, 80, 100, 200, 300, 400, 540, 800, 910, 1020, 1282, 1322, 1342, 1352, 1362, 1372, 1382, 1392, 1402, 1412, 1422, 1442, 1472, 1502, 1512, 1522, 1532, 1542, 1552, 1562, 1572]

y, label, pots = drawBasic([0.0126, 0.0071, 0.00315, 0.00084], x, [1, 0.75, 0.5, 0.25])
print(y)

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)
ax.axis([0, 1500, 0, 1300])

line_sym = ["-", "--", "-.", "-"]
pot_sym = ["o", ">", "<", "x"]
colors = ['blue', 'brown', 'aqua', 'green']

for i in range(len(y)):
     ax.plot(x, y[i],line_sym[i], marker=pot_sym[i], ms=6, label=label[i], linewidth=2, color=colors[i])

y_sche = [79.36507936507937, 79.36507936507937, 80.90207914151576, 108.87547507265816, 235.33422758774873, 309.8069900886802, 564.0873015873016, 808.9359200470311, 1036.6071428571427, 1093.5810221524505, 1138.2664176781823, 1105.0442017680707, 1071.6086737266767, 1055.6383507203177, 1047.8303747534517, 1040.1370533529123, 1032.5558794946548, 1025.0844187168354, 1017.7203065134099, 1010.4612458392771, 1003.3050047214351, 996.2494139709328, 982.431807674526, 962.409420289855, 943.1868619618286, 936.9488536155203, 930.7928164695575, 924.7171453437771, 918.7202766969303, 912.8006872852234, 906.9568928723858, 901.1874469889736]
ax.plot(x, y_sche, "--", marker='*', ms=10, label="combine of sub-models", linewidth=2)


for k in range(len(pots)):
     ele = pots[k]
     for i in range(len(ele)):
          xc1, yc1 = ele[i]
          ax.plot(xc1, yc1, line_sym[k], marker=pot_sym[k], ms=10, color=colors[k])


plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel('Workload(#examples/second)', fontsize=20)
ax.set_ylabel('Effective speed', fontsize=20)
plt.grid(linestyle='-.')
plt.savefig('./Speed.jpg')
plt.show()
# 每妙发送500条








