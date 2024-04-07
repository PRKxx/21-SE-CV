import cv2
import numpy as np
import matplotlib.pyplot as plt

# 将附带的彩色图像转化为灰度图像

image = cv2.imread(r'.\assert\I0.jpg')

# 1.计算公式 Gray = 0.3R + 0.59G + 0.11B
image_gray = image[:, :, 0] * 0.11 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.3
image_gray = image_gray.astype(np.uint8)

# 2.cv库自动转化为灰度图
image_gray_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("assert/I1.jpg", image_gray_cv2)
