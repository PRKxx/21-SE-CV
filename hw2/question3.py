import cv2
import numpy as np
from scipy.signal import convolve2d


def mean_filter(image, kernel_size):
    # 均值滤波器，在一个kernel内取这些像素点的平均值，然后滑动
    height, width = image.shape[:2]
    filtered_image = np.zeros_like(image)
    kernel_half = kernel_size // 2
    for i in range(height):
        for j in range(width):
            # 确定其实窗口位置，保证不超出图片范围
            x_start = max(0, i - kernel_half)
            x_end = min(height, i + kernel_half + 1)
            y_start = max(0, j - kernel_half)
            y_end = min(width, j + kernel_half + 1)

            window = image[x_start:x_end, y_start:y_end]
            # 求平均值
            filtered_image[i, j] = np.mean(window)

    return filtered_image.astype(np.uint8)


def median_filter(image, kernel_size):
    # 中值滤波器，在一个kernel内取这些像素点的中值，然后滑动
    height, width = image.shape[:2]
    filtered_image = np.zeros_like(image)
    kernel_half = kernel_size // 2
    for i in range(height):
        for j in range(width):
            x_start = max(0, i - kernel_half)
            x_end = min(height, i + kernel_half + 1)
            y_start = max(0, j - kernel_half)
            y_end = min(width, j + kernel_half + 1)

            window = image[x_start:x_end, y_start:y_end].ravel()
            # 求中值
            filtered_image[i, j] = np.median(window)

    return filtered_image.astype(np.uint8)


# 高斯函数 G(X, Y)
def gaussian(x, y, sigma):
    return 1 / (2 * np.pi * sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))


def gaussian_filter(image, size=5, sigma=1.0):
    # 高斯滤波器, 高斯滤波器的核心思想是根据高斯函数计算每个像素点的权重。
    # 距离中心像素越远的像素，其权重越小。这样可以使得滤波器更多地关注周围像素，减少噪声的影响
    tmp = np.zeros((size, size))
    center = size // 2
    for x in range(size):
        for y in range(size):
            tmp[x, y] = gaussian(x - center, y - center, sigma)
    # 计算好了kernel中的每一个像素的权重值
    kernel = tmp / np.sum(tmp)
    # 卷积运算较为复杂
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image


image = cv2.imread(r'.\assert\noise_random.jpg')
# mean_filter = mean_filter(image, 5)
# median_filter = median_filter(image, 5)
# gaussian_filter = gaussian_filter(image)
mean_filter = cv2.blur(image, (3, 3))
cv2.imwrite("assert/mean_filter.jpg", mean_filter)
median_filter = cv2.medianBlur(image, 3)
cv2.imwrite("assert/median_filter.jpg", median_filter)
gaussian_filter = cv2.GaussianBlur(image, (3, 3), 2)
cv2.imwrite("assert/gaussian_filter.jpg", gaussian_filter)