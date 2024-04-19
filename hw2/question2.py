import random

import cv2
import numpy as np


def random_noise(image, noise_num):
    # 添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）

    img = cv2.imread(image)
    img_noise = img
    rows, cols, chn = img_noise.shape
    for i in range(noise_num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise


def sp_noise(image, prob):
    # 添加椒盐噪声，椒盐噪声是指两种噪声，一种是盐噪声，另一种是椒噪声。
    # 盐=白色(0)，椒=黑色(255)。前者是高灰度噪声，后者属于低灰度噪声。
    # 一般两种噪声同时出现，呈现在图像上就是黑白杂点

    image = cv2.imread(image)
    output = np.zeros(image.shape, np.uint8)
    tmp = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > tmp:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    result = output
    return result


def Gauss_noise(image, mean=0, var=0.001):
    # 添加高斯噪声，方差越大，噪声越明显

    image = cv2.imread(image)

    # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    image = np.array(image / 255, dtype=float)
    # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    # 将噪声和原始图像进行相加得到加噪后的图像
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    # clip函数将元素的大小限制在了low_clip和1之间了，小于low_clip的用low_clip代替，大于1的用1代替
    out = np.clip(out, low_clip, 1.0)
    # 解除归一化，乘以255将加噪后的图像的像素值恢复
    out = np.uint8(out * 255)
    return out


noise_Gauss = Gauss_noise("./assert/I1.jpg", mean=0, var=0.05)
noise_random = random_noise("./assert/I1.jpg", 100000)
noise_sp = sp_noise("./assert/I1.jpg", 0.1)
cv2.imwrite("assert/noise_Gauss.jpg", noise_Gauss)
cv2.imwrite("assert/noise_random.jpg", noise_random)
cv2.imwrite("assert/noise_sp.jpg", noise_sp)
