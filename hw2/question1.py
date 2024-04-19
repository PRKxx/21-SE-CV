import cv2

# 将附带的彩色图像转化为灰度图像
image = cv2.imread(r'.\assert\I0.jpg')
# cv库自动转化为灰度图
image_gray_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("assert/I1.jpg", image_gray_cv2)
