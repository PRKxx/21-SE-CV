import cv2
import numpy as np
from PIL import Image


def merge_images(output_path, *image_paths):
    images = [Image.open(path) for path in image_paths]
    width, height = images[0].size
    padding = 20
    new_width = width * 2 + padding
    new_height = height * 4 + padding * 3
    new_image = Image.new('RGB', (new_width, new_height), color='white')
    for i, img in enumerate(images):
        x = (i % 2) * (width + padding)
        y = (i // 2) * (height + padding)
        new_image.paste(img, (x, y))
    new_image.save(output_path)


# 方法：通过将灰度照片进行与操作来提取每一位上的值，再进行替换
I1 = cv2.imread('./assert/I1.jpg')
I2 = cv2.imread('./assert/I2.jpg')
binary_images = [np.zeros_like(I1) for _ in range(8)]
for i in range(8):
    binary_images[i] = I1 & (1 << i)
    binary_images[i] = np.uint8(binary_images[i])
for i in range(8):
    binary_images[i] = np.where(binary_images[i] > 0, I2, binary_images[i])
    cv2.imwrite(f"binary_image_{i}.png", binary_images[i])

# 合并 8 张图片
image_paths = ['binary_image_{}.png'.format(i) for i in range(8)]
merge_images('./assert/merge_img.png', *image_paths)
