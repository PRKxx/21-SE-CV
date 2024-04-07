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


image = cv2.imread('./assert/I0.jpg')

# 亮度变换
for i in range(8):
    tmp = cv2.convertScaleAbs(image, beta=(-120 + i * 30))
    cv2.imwrite(f"light{i}_img.png", tmp)
image_paths = ['light{}_img.png'.format(i) for i in range(8)]
merge_images('./assert/light_img.png', *image_paths)

# 对比度变换
for i in range(8):
    tmp = cv2.convertScaleAbs(image, alpha=(0 + i * 0.25))
    cv2.imwrite(f"contract{i}_img.png", tmp)
image_paths = ['contract{}_img.png'.format(i) for i in range(8)]
merge_images('./assert/contract_img.png', *image_paths)

# 饱和度变化
for i in range(8):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hls[:, :, 2] = np.clip(30 * i, 0, 255).astype(np.uint8)
    tmp = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    cv2.imwrite(f"saturation{i}_img.png", tmp)
image_paths = ['saturation{}_img.png'.format(i) for i in range(8)]
merge_images('./assert/saturation_img.png', *image_paths)
