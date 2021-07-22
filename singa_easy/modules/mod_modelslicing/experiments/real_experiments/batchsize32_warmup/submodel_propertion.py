# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


scheduler_map = \
    {32: [32, 0, 0, 0], 160: [160, 0, 0, 0], 320: [320, 0, 0, 0], 640: [640, 0, 0, 0], 960: [960, 0, 0, 0],
     1280: [1280, 0, 0, 0], 1600: [1600, 0, 0, 0], 1920: [1920, 0, 0, 0], 2240: [2240, 0, 0, 0], 2560: [2560, 0, 0, 0],
     3200: [3200, 0, 0, 0], 3840: [3840, 0, 0, 0], 4480: [4480, 0, 0, 0], 4800: [4800, 0, 0, 0], 5120: [5120, 0, 0, 0],
     5440: [5440, 0, 0, 0], 5760: [5586, 0, 174, 0], 6400: [4937, 0, 1463, 0], 8000: [3314, 0, 4686, 0],
     9600: [1691, 0, 7909, 0], 11200: [68, 1, 11131, 0], 12800: [0, 0, 7854, 4946], 17600: [0, 0, 1, 16325],
     19200: [0, 0, 1, 16325]}


name_list = list(scheduler_map.keys()).sort()
# name_list = [32, 3200, 9600, 19200, 25600]


fig = plt.figure(figsize=(11, 10))
ax = fig.add_subplot(111)
ax.axis([0, len(list(scheduler_map.keys())), 0, 17500])

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
ax.bar(range(len(y100)), y100, label='Sub-model with $r_i$ = 1', tick_label=name_list, fc=colors[0])
ax.bar(range(len(y75)), y75, bottom=y100, label='Sub-model with $r_i$ = 0.75', tick_label=name_list, fc=colors[1])
ax.bar(range(len(y5)), y5, bottom=y75, label='Sub-model with $r_i$ = 0.5', tick_label=name_list, fc=colors[2])
ax.bar(range(len(y025)), y025, bottom=y5, label='Sub-model with $r_i$ = 0.25', fc=colors[3])


plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

ax.set_xlabel('Ingesting rate(#examples/second)', fontsize=20)
ax.set_ylabel('#Examples', fontsize=20)

plt.grid(linestyle='-.')
plt.savefig('./sub_model_proportion.jpg')
plt.show()
