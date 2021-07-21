










import matplotlib.pyplot as plt
import numpy


def drawBasic(tlist, x, slicerate):

     y = []
     label = []
     pots = []

     for k in range(len(tlist)):
          interval0 = [1/tlist[k] for r in x]
          y1 = interval0
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


# x_5s = [1, 50, 80, 100, 200, 300, 400, 540, 800, 910, 1020, 1282, 1342, 1352, 1362, 1372, 1382, 1392, 1402, 1412, 1422, 1442, 1472, 1502, 1512, 1522, 1532, 1542, 1552, 1572]
x = [1, 50, 80, 100, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2400, 2500, 2600, 2700, 2800, 3000, 3200, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4300, 4700, 5100, 6000]


y, label, pots = drawBasic([0.0126, 0.0071, 0.00315, 0.00084], x, [1, 0.75, 0.5, 0.25])
print(y)

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)
ax.axis([0, 5500, 0, 1650])

line_sym = ["-", "--", "-.", "-"]
pot_sym = ["o", ">", "<", "x"]
colors = ['blue', 'brown', 'aqua', 'green']

for i in range(len(y)):
     ax.plot(x, y[i],line_sym[i], marker=pot_sym[i], ms=10, label=label[i], linewidth=2, color=colors[i])

y_sche = [79.36507936507937, 79.36507936507937, 79.36507936507937, 79.36507936507937, 79.36507936507937, 79.36507936507937, 79.36507936507935, 100.12515644555694, 125.14078338130396, 150.02625459455405, 175.07440662281468, 200.1350911865509, 225.2083176938668, 250.04688379071075, 275.1203651597574, 300.20639189442744, 350.103280467738, 375.03187770960534, 400.18108193957767, 425.1030874987185, 450.01462547532793, 475.1900760304122, 500.0950180534302, 525.2928507643011, 600.3031530923116, 625.1844294066749, 650.0552546966492, 675.3055757730373, 700.1697911743597, 750.3001200480192, 800.4462487836969, 875.442098259621, 900.265578345612, 925.0786316836931, 950.4300696064969, 975.2364948500011, 1000.0325010562843, 1025.4101640656263, 1075.5996468030926, 1175.346727284549, 1190.4761904761904, 1190.4761904761904]
ax.plot(x, y_sche, "--", marker='*', ms=10, label="combination of sub-models", linewidth=2)


plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel('Ingesting rate(#examples/second)', fontsize=20)
ax.set_ylabel('Throughput(examples processed/second)', fontsize=20)
plt.grid(linestyle='-.')
plt.savefig('./throughput.jpg')
plt.show()
# 每妙发送500条
