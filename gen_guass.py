import numpy as np
from PIL import Image

# 生成高斯噪声数据，每个颜色通道都有高斯噪声
mean = 128  # 均值
stddev = 128  # 标准差
image_size = (32, 32, 3)  # 图像大小，包括三个颜色通道

gaussian_noise = np.random.normal(mean, stddev, image_size)

# 将数据限制在0到255的范围内（RGB图像的像素值范围）
gaussian_noise = np.clip(gaussian_noise, 0, 255).astype(np.uint8)

# 创建PIL图像对象
image = Image.fromarray(gaussian_noise)

# 保存图像到文件
image.save('pic.png')
