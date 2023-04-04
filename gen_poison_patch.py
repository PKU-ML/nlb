import numpy as np
import torch
import torchvision 

# c, h, w = 3, 32, 32
# a = np.zeros([3, 32, 32]).astype(np.uint8)
# for i in range(h):
#     for j in range(w):
#         if (i + j) % 2 == 0:
#             a[:, i, j] = 255

# torchvision.io.write_jpeg(torch.from_numpy(a), 'data/checkboard.jpg')

mean=(0.485, 0.456, 0.406)
std=(0.228, 0.224, 0.225)

c, h, w = 3, 1000, 1000
mean = torch.tensor(mean).view(3, 1, 1)
std = torch.tensor(std).view(3, 1, 1)
a = torch.randn(c,h,w) * std + mean
a.clamp_(0.0, 1.0)
a = (a * 255).type(torch.uint8)

torchvision.io.write_jpeg(a, 'data/gaussian_in.jpg')
