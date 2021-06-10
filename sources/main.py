import numpy as np
from PIL import Image, ImageDraw
from math import sqrt
import glob
import matplotlib.pyplot as plt
from image_properties import ImagePreProcessing

# Load all images in the folder 'Image folder'
image_list = []
im_array = []
for filename in glob.glob(r'./../Image folder/*.bmp'):
    im = Image.open(filename)
    image_list.append(im)

# Make arrays from the BMP information
for i in range(len(image_list)):
    im_array.append(np.array(image_list[i]))
print(np.shape(im_array)[1])
print(np.shape(im_array)[2])
print(im_array[1][0][0])

input_pixels = image_list[0].load()
width, height = image_list[0].width, image_list[0].height
for i in range(len(image_list)):
    print('Image {} has the following information:'.format(i))
    print(image_list[i].format, image_list[i].size, image_list[i].mode)

print(np.shape(image_list[0]))
# Create output image
output_image = Image.new("RGB", image_list[0].size)
draw = ImageDraw.Draw(output_image)

if len(np.shape(image_list[0])) == 3:
    # convert to grayscale
    intensity = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            intensity[x, y] = sum(input_pixels[x, y]) / 3

    # Compute convolution between intensity and kernels
    for x in range(1, image_list[0].width - 1):
        for y in range(1, image_list[0].height - 1):
            magx = intensity[x+1, y] - intensity[x-1, y]
            magy = intensity[x, y + 1] - intensity[x, y - 1]

            # Draw in black and white the magnitude
            color = int(sqrt(magx**2 + magy**2))
            draw.point((x, y), (color, color, color))
    # output_image.save(r'./../Image folder/output.bmp')

# Create difference Matrix from 2 or more Images
diff_matrix_1 = np.zeros_like(im_array[0])
diff_matrix_2 = np.zeros_like(im_array[0])
for i in range(0, np.shape(im_array)[1]):
    for j in range(0, np.shape(im_array)[2]):
        diff_matrix_1[i][j] = abs(im_array[0][i][j] - im_array[1][i][j])
        diff_matrix_2[i][j] = abs(im_array[0][i][j] - im_array[4][i][j])

print(diff_matrix_1)

plt.figure()
plt.imshow(diff_matrix_2, cmap='gray')
plt.show()