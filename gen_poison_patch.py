import numpy as np
import torch
import torchvision 

c, h, w = 3, 32, 32
a = np.zeros([3, 32, 32]).astype(np.uint8)
for i in range(h):
    for j in range(w):
        if (i + j) % 2 == 0:
            a[:, i, j] = 255

torchvision.io.write_jpeg(torch.from_numpy(a), 'data/checkboard.jpg')
