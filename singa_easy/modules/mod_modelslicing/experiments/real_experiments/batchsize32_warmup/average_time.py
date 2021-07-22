import random

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np



def drawBasic(tlist, x, slicerate):

     y = []
     label = []
     pots = []

     a = [0.5445709824562073, 0.5094479918479919, 0.5183689951896667, 0.5360075980424881, 0.5292132049798965,
          0.4437957026064396, 0.47011234015226366, 0.5176802704731623, 0.48556351321084157, 0.48278466146439314,
          0.44622810930013657, 0.4448867492377758, 0.46625482248408456, 0.43684681246678037, 0.4495523855090141,
          0.4541638591710259, 0.44788666591048243, 0.46880729161202905, 0.46132540863752364, 0.4416423341135184,
          0.428506940305233, 0.4360292255505919, 0.4623933795425627, 0.433588912405751, 0.44642280504107473,
          0.44519318903868016, 0.43057115663375173, 0.4433484307726224, 0.44249152136966585, 0.43079319958388806]
     b = [0.6915900111198425, 0.7784101963043213, 0.6521405041217804, 0.8237374037504196, 0.8259381254514059,
          0.7572330482304096, 0.7212126004695892, 0.7194328308105469, 0.7206863846097674, 0.6823824867606163,
          0.6890312218666077, 0.6694948720435302, 0.6514000279562814, 0.6753530846039454, 0.6769931729882955,
          0.6651239747510237, 0.656006221804354, 0.6626777055859566, 0.6443936984539032, 0.6338313501079877,
          0.6470637729338237, 0.6443282450735569, 0.6400967982742521, 0.6373766519806602, 0.6468611674010754,
          0.6315480814530299, 0.6350046985915729, 0.6442887659072876, 0.6393341201171279, 0.6366256182193756]
     c = [1.1352440118789673, 1.0711206078529358, 1.0130944073200225, 1.2538503527641296, 1.1771859387556711,
          1.1115223228931428, 1.0740656352043152, 1.066507320602735, 1.0453771433659962, 1.0237999111413956,
          1.0099865812063218, 0.9922433346509933, 1.0077772610953877, 1.0082984054088593, 0.9782136220484972,
          0.9860923223635729, 0.9577391134368048, 0.999786058217287, 0.9853356564044953, 0.9738650581240654,
          0.9787426404442106, 0.9775253287702799, 0.987561702132225, 0.9740239739959891, 0.9382529258231322,
          0.9649664891683138, 0.9326017934509686, 0.980001263697942, 0.9518248564377427, 0.9663746791481972]
     d = [1.5324100255966187, 1.3823970079421997, 1.374810791015625, 1.7053762376308441, 1.5713075002034504,
          1.5109713420271873, 1.4090474152565002, 1.3475713342428208, 1.3517827570438385, 1.325958064198494,
          1.2840893298387528, 1.27940504103899, 1.2620532372168132, 1.2815734720230103, 1.2848724655807018,
          1.3092085389529957, 1.2745403432183795, 1.2426788526773453, 1.30669841837883, 1.220376658042272,
          1.2926877955027989, 1.2330426709353923, 1.2477019797431097, 1.219258243820884, 1.2797774475812913,
          1.2148992225757012, 1.22192884351526, 1.2778068358898163, 1.2740839910507202, 1.2091002200841903]
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
ax.axis([0, 1000*32, 0.25, 1.8])

line_sym = ["-", "--", "-.", "-"]
pot_sym = ["o", ">", "<", "x"]
colors = ['blue', 'brown', 'aqua', 'green']
for i in range(len(y)):
     y_i = [ele for ele in y[i]]
     ax.plot(x, y_i, line_sym[i], marker=pot_sym[i], ms=10, label=label[i], linewidth=2,  color=colors[i])


plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xlabel('Ingesting rate(#examples/second)', fontsize=20)
ax.set_ylabel('Inference time per example(millisecond)', fontsize=20)
plt.grid(linestyle='-.')
plt.savefig('./average_time.jpg')
plt.show()






