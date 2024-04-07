import cv2
import numpy as np

# 将手写的图片转化为二值图像

img = cv2.imread("./assert/handwrite.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 方法1：将图片转化为灰度图像，并设置阈值，如果超过这个阈值就是黑色的，反之是白色的
threshold = 127
ret, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

# 方法2：使用自适应转化，更加灵活
binary_ = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imwrite("assert/I2.jpg", binary, [cv2.IMWRITE_PNG_COMPRESSION, 0])


