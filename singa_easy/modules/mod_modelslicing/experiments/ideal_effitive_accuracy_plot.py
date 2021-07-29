
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1.2, 300)
print(x)


interval0 = [0.95 if (i < 0.25) else 0 for i in x]
interval1 = [0.95 if (i > 0.25) else 0 for i in x]
y1 = interval0 + (0.25/x) * interval1

interval2 = [0.8 if (i < 0.5) else 0 for i in x]
interval3 = [0.8 if (i > 0.5) else 0 for i in x]
y2 = interval2 + (0.5/x) * interval3

interval4 = [0.75 if (i < 1) else 0 for i in x]
interval5 = [0.75 if (i > 1) else 0 for i in x]
y3 = interval4 + (1/x) * interval5


interval01 = [0.9 if (i < 100) else 0 for i in x]
interval02 = [1 if (100 < i <130) else 0 for i in x]
interval03 = [0.7 if (200 >= i > 130) else 0 for i in x]
interval04 = [1 if (200 < i < 235) else 0 for i in x]


interval05 = [0.6 if ( 235 < i < 300) else 0 for i in x]
interval06 = [1 if (i > 300) else 0 for i in x]

interval0000 = [0.05 for i in x]

y4 = interval01 + \
     (90/x) * interval02 + \
     interval03 + \
     (140/x) * interval04 + \
     interval05 + \
     (180/x) * interval06 + interval0000


plt.xlim((0, 1.2))
plt.ylim((0, 1))
plt.xlabel('$W_{exp}=N/D$ (# instances / deadline constraint)', fontsize=15)
plt.ylabel('Effective accuracy %', fontsize=15)


plt.plot(x, y1, label='Sub-model with accuracy 0.95')
plt.plot(x, y2, label='Sub-model with accuracy 0.8')
plt.plot(x, y3, label='Sub-model with accuracy 0.78')
# plt.plot(x, y4, label='list4')

# 画点和线
plt.vlines(0.25, 0, 1, colors="c", linestyles="dashed")
plt.vlines(0.5, 0, 1, colors="c", linestyles="dashed")
plt.vlines(1, 0, 0.8, colors="c", linestyles="dashed")
plt.vlines(0.3, 0, 1, colors="c", linestyles="dashed")
plt.vlines(0.53333, 0, 1, colors="c", linestyles="dashed")

xc1, yc1 = 0.25, 0.95
plt.scatter(xc1, yc1, s=80, facecolors='none', edgecolors='r')
xc1, yc1 = 0.3, 0.8
plt.scatter(xc1, yc1, s=80, facecolors='none', edgecolors='r')
xc1, yc1 = 0.5, 0.8
plt.scatter(xc1, yc1, s=80, facecolors='none', edgecolors='r')
xc1, yc1 = 0.533, 0.75
plt.scatter(xc1, yc1, s=80, facecolors='none', edgecolors='r')
xc1, yc1 = 1, 0.75
plt.scatter(xc1, yc1, s=80, facecolors='none', edgecolors='r')

plt.legend()
plt.savefig('./effective accuracy.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()

# CUDA_VISIBLE_DEVICES=0,1,2 python train.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 100 --batch_size 256 --lr 0.1  --dataset cifar10 --data_dir $HOME/data/ --log_freq 50
