# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

scheduler_map = {1: [1, 0, 0, 0],
                 5: [5, 0, 0, 0],
                 10: [10, 0, 0, 0],
                 20: [20, 0, 0, 0],
                 30: [30, 0, 0, 0],
                 40: [40, 0, 0, 0],
                 50: [50, 0, 0, 0],
                 60: [60, 0, 0, 0],
                 70: [70, 0, 0, 0],
                 80: [80, 0, 0, 0],
                 100: [100, 0, 0, 0],
                 120: [120, 0, 0, 0],
                 140: [140, 0, 0, 0],
                 150: [150, 0, 0, 0],
                 160: [160, 0, 0, 0],
                 170: [170, 0, 0, 0],
                 180: [180, 0, 0, 0],
                 200: [154, 0, 46, 0],
                 250: [93, 10, 146, 0],
                 300: [51, 0, 246, 3],
                 350: [2, 0, 348, 0],
                 400: [0, 0, 245, 155],
                 450: [0, 0, 134, 316],
                 500: [0, 0, 22, 478],
                 550: [0, 0, 0, 550],
                 600: [0, 0, 0, 600],
                 650: [0, 0, 0, 650],
                 700: [0, 0, 0, 700],
                 750: [0, 0, 0, 750],
                 800: [0, 0, 0, 800]}


# scheduler_map = {
#                  180: [180, 0, 0, 0],
#                  200: [154, 0, 46, 0],
#                 }

name_list = list(scheduler_map.keys()).sort()


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
x = list(scheduler_map.keys())

x = [ele*32 for ele in x]

ax.axis([1, 1000*32, 0, 800])

y025 = []
y5 = []
y75 = []
y100 = []

for key in scheduler_map:
    y025.append(scheduler_map[key][3])
    y5.append(scheduler_map[key][2])
    y75.append(scheduler_map[key][1])
    y100.append(scheduler_map[key][0])

colors = ['lightskyblue', 'brown', 'aqua', 'plum']
ax.bar(x, y100, label='Sub-model with $r_i$ = 1', tick_label=name_list, fc=colors[0], width=20*32)
ax.bar(x, y75, bottom=np.array(y100), label='Sub-model with $r_i$ = 0.75', tick_label=name_list, fc=colors[1], width=20*32)
ax.bar(x, y5, bottom=np.array(y100)+np.array(y75), label='Sub-model with $r_i$ = 0.5', tick_label=name_list, fc=colors[2], width=20*32)
ax.bar(x, y025, bottom=np.array(y100)+np.array(y75)+np.array(y5), label='Sub-model with $r_i$ = 0.25', fc=colors[3], width=20*32)

plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

ax.set_xlabel('Ingesting rate(#instances/second)', fontsize=20)
ax.set_ylabel('# Mini-batches', fontsize=20)

plt.grid(linestyle='-.')
plt.savefig('./pdfs/sub_model_rate.pdf', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.1)
plt.show()
