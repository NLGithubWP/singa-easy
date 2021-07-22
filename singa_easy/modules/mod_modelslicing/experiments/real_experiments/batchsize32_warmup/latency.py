import random

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np

a = [17.426271438598633, 81.51167869567871, 165.87807846069336, 343.0448627471924, 508.0446767807007,
     568.0584993362427, 752.1797442436218, 993.9461193084717, 1087.6622695922852, 1235.9287333488464,
     1427.929949760437, 1708.365117073059, 2088.8216047286987, 2096.8646998405457, 2301.7082138061523,
     2470.651393890381, 2579.8271956443787, 3000.366666316986, 3690.603269100189, 4239.766407489777,
     4799.27773141861, 5581.174087047577, 6658.464665412903, 7631.164858341217, 8571.317856788635,
     9260.018332004547, 9644.793908596039, 10640.362338542938, 11327.782947063446, 13785.382386684418]
b = [22.13088035583496, 124.5456314086914, 208.68496131896973, 527.1919384002686, 792.9006004333496,
     969.2583017349243, 1153.9401607513428, 1381.31103515625, 1614.337501525879, 1746.8991661071777,
     2204.8999099731445, 2570.8603086471558, 2918.2721252441406, 3241.694806098938, 3466.2050457000732,
     3618.274422645569, 3778.5958375930786, 4241.137315750122, 5155.149587631226, 6084.780961036682,
     7247.114256858826, 8247.401536941528, 9217.393895149231, 11217.82907485962, 12419.734414100647,
     13136.200094223022, 14224.105248451233, 15462.930381774902, 16366.953474998474, 20372.01978302002]
c = [36.32780838012695, 171.37929725646973, 324.1902103424072, 802.464225769043, 1130.0985012054443,
     1422.7485733032227, 1718.5050163269043, 2047.694055557251, 2341.6448011398315, 2620.9277725219727,
     3231.9570598602295, 3810.2144050598145, 4514.842129707336, 4839.832345962524, 5008.453744888306,
     5364.342233657837, 5516.577293395996, 6398.630772590637, 7882.685251235962, 9349.104557991028,
     10961.917572975159, 12512.324208259583, 14220.88851070404, 17142.821942329407, 18014.45617580414,
     20071.302974700928, 20890.280173301697, 23520.03032875061, 24366.716324806213, 30923.98973274231]
d = [49.0371208190918, 221.18352127075195, 439.939453125, 1091.4407920837402, 1508.4552001953125,
     1934.0433177947998, 2254.4758644104004, 2587.336961746216, 3027.9933757781982, 3394.4526443481445,
     4109.085855484009, 4912.915357589722, 5653.998502731323, 6151.552665710449, 6578.547023773193,
     7122.094451904297, 7341.352376937866, 7953.14465713501, 10453.58734703064, 11715.61591720581,
     14478.103309631348, 15782.946187973022, 17966.90850830078, 21458.94509124756, 24571.72699356079,
     25269.903829574585, 27371.20609474182, 30667.36406135559, 32616.550170898438, 38691.20704269409]

def drawBasic(tlist, x, slicerate):

     y = []
     label = []
     pots = []


     y.append(d)
     y.append(c)
     y.append(b)
     y.append(a)

     for k in range(len(tlist)):

          label.append('Sub-model with $r_i$ = ' + str(slicerate[k]))

          tmp = []
          for i in range(len(x)):
               tmp.append([x[i], y[k][i]])
          pots.append(tmp)

     return y, label, pots

import numpy


x = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 150, 160, 170, 180, 200, 250, 300, 350, 400, 450, 550, 600, 650, 700, 750, 800, 1000]
x = numpy.array(x)
x = [ele*32 for ele in x]
y, label, pots = drawBasic([0.00141, 0.00110, 0.00071, 0.00049], x, [1, 0.75, 0.5, 0.25])


fig=plt.figure(figsize=(10, 10))
ax=fig.add_subplot(111)
ax.axis([0, 1000*32, 0, 20])

line_sym = ["-", "--", "-.", "-"]
pot_sym = ["o", ">", "<", "x"]
colors = ['blue', 'brown', 'aqua', 'green']
for i in range(len(y)):
     y_i = [ele/1000 for ele in y[i]]
     ax.plot(x, y_i, line_sym[i], marker=pot_sym[i], ms=10, label=label[i], linewidth=2,  color=colors[i])


y_idealy = [0.04512, 0.2256, 0.4512, 0.9024, 1.3536, 1.8048, 2.2560000000000002, 2.7072, 3.1584, 3.6096, 4.5120000000000005, 5.4144, 6.3168, 6.768, 7.2192, 7.6704, 7.9998000000000005, 7.9999, 7.9998000000000005, 7.999700000000001, 7.99999, 7.99988, 7.99988,  7.9999, 9.40822, 10.192219999999999, 10.97622, 11.76022, 12.54422, 15.680219999999998]
# ysch = [a[i]/1000+random.randrange(-5, 5)*0.1 if (x[i]>17900) else ysch[i] for i in range(len(x))]

y_real_experiment = [50.54902267456055, 242.35004425048828, 459.1717109680176, 1547.4368743896484, 1266.5564231872559, 1668.409954071045, 2094.574909210205, 2514.7609519958496, 3480.23384475708, 3337.1972732543945, 4807.21407699585, 4955.72673034668, 5869.175651550293, 6753.157760620117, 6571.647644042969, 7063.424160003662, 7263.424160003662, 7397.05934715271, 7394.448558807373, 7165.690972328186, 7844.754432678223, 7320.80154800415, 7294.447976112366, 8148.598042488098, 8862.68032169342, 10567.460627555847, 10400.686946868896, 10543.084606170654, 11596.354914665222, 14493.859671592712]
y_real_experiment = [ele/1000 for ele in y_real_experiment]

ax.plot(x, y_real_experiment, "--", marker='*', ms=15, label="Combination of sub-models", linewidth=4)
ax.plot(x, y_idealy, "--", label="Theoretical value", linewidth=4)


plt.hlines(8.1, 0, 85000, label="Deadline constrain=8s", linewidth=4, color="red")
#
plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel('Ingesting rate(#examples/second)', fontsize=20)
ax.set_ylabel('Latency(second)', fontsize=20)
plt.grid(linestyle='-.')
plt.savefig('./latency.jpg')
plt.show()








