import cv2

from hw2.question2 import Gauss_noise, random_noise, sp_noise

noise_Gauss = Gauss_noise("./assert/I0.jpg", mean=0, var=0.05)
noise_random = random_noise("./assert/I0.jpg", 100000)
noise_sp = sp_noise("./assert/I0.jpg", 0.1)
cv2.imwrite("assert/noise_Gauss_I0.jpg", noise_Gauss)
cv2.imwrite("assert/noise_random_I0.jpg", noise_random)
cv2.imwrite("assert/noise_sp_I0.jpg", noise_sp)

img = noise_sp
mean_filter = cv2.blur(img, (3, 3))
cv2.imwrite("assert/mean_filter_I0.jpg", mean_filter)
median_filter = cv2.medianBlur(img, 3)
cv2.imwrite("assert/median_filter_I0.jpg", median_filter)
gaussian_filter = cv2.GaussianBlur(img, (3, 3), 2)
cv2.imwrite("assert/gaussian_filter_I0.jpg", gaussian_filter)
