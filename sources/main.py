import numpy as np
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from canny import canny_edge_detector
from collections import defaultdict
import glob
import cv2
import matplotlib.pyplot as plt


# Load all images in the folder 'Image folder'
image_list = []
im_array = []
for filename in glob.glob(r'./../Image folder/*.bmp'):
    im = Image.open(filename)
    image_list.append(im)

# Make arrays from the BMP information
for i in range(len(image_list)):
    im_array.append(np.array(image_list[i], dtype='int64'))
print(np.shape(im_array)[1])
print(np.shape(im_array)[2])
print(im_array[1][0][0])

input_pixels = image_list[0].load()
width, height = image_list[0].width, image_list[0].height
for i in range(len(image_list)):
    print('Image {} has the following information:'.format(i))
    print(image_list[i].format, image_list[i].size, image_list[i].mode)

print(np.shape(image_list[0]))

# Create difference Matrix from 2 or more Images
diff_matrix = np.zeros_like(im_array[0])
for i in range(0, np.shape(im_array)[1]):
    for j in range(0, np.shape(im_array)[2]):
        diff_matrix[i][j] = abs(im_array[0][i][j] - im_array[4][i][j])
        if 100 > diff_matrix[i][j]:
            diff_matrix[i][j] = 0
        if diff_matrix[i][j] > 180:
            diff_matrix[i][j] = 255

plt.figure()
plt.imshow(diff_matrix, cmap='gray')
plt.savefig(r'./../Output folder/test.png')
plt.show()
print(diff_matrix)
print(np.max(diff_matrix), np.min(diff_matrix))

# Load image:
#input_image = Image.open("input.png")
input_image = Image.fromarray(diff_matrix, 'LA')

# Output image:
output_image = Image.new("LA", input_image.size)
output_image.paste(input_image)
plt.imshow(output_image, cmap='gray')
plt.show()
draw_result = ImageDraw.Draw(output_image)

# Find circles
rmin = 18
rmax = 20
steps = 100
threshold = 0.4

points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

acc = defaultdict(int)
for x, y in canny_edge_detector(input_image):
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1

circles = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
        print(v / steps, x, y, r)
        circles.append((x, y, r))

for x, y, r in circles:
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255, 0, 0, 0))

