
from torchvision.models import resnet50
import torch.nn as nn
import os

from shutil import move

base = "/Users/nailixing/Downloads/data/"

dirs = os.listdir(base+"food_mini")
print(len(dirs))

#
# for d in dirs:
#     os.mkdir(base + "food_mini_test/" + d)
#
#
# for d in dirs:
#     if d == ".DS_Store":
#         continue
#     i = 0
#     for img in os.listdir(base+"food_mini/"+d):
#         move(base+"food_mini/"+d+"/"+img, base+"food_mini_test/"+d+"/"+img)
#         i += 1
#         if i > 10:
#             break
#

