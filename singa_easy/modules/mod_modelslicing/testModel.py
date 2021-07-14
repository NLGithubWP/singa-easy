

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


a = [[False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False,  True, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False,  True, False, False, False, False, False, False,
         False, False, False, False],
        [False, False,  True, False, False, False, False, False, False, False,
         False, False, False, False, False,  True, False, False, False, False,
         False, False, False, False, False, False, False, False,  True, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False,  True, False, False, False, False,
         False, False, False, False, False, False,  True, False, False, False,
         False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
          True, False, False, False, False, False, False, False, False, False,
         False,  True, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,  True,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False]]

c = [
        [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
        [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51],
        [23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8]
]

t = [
      [44, 15, 51, 50, 36, 17, 30, 32, 46, 17, 12, 20, 25, 14, 46, 51, 37,  4],
      [44, 15, 51, 50, 36, 17, 30, 32, 46, 17, 12, 20, 25, 14, 46, 51, 37,  4],
      [44, 15, 51, 50, 36, 17, 30, 32, 46, 17, 12, 20, 25, 14, 46, 51, 37,  4],
      [44, 15, 51, 50, 36, 17, 30, 32, 46, 17, 12, 20, 25, 14, 46, 51, 37,  4],
      [44, 15, 51, 50, 36, 17, 30, 32, 46, 17, 12, 20, 25, 14, 46, 51, 37,  4]
]

b = torch.Tensor(a)
print(b, b[:3].size())
