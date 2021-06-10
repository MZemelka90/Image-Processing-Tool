import numpy as np
from PIL import Image, ImageDraw
from math import sqrt
import glob
from image_properties import ImagePreProcessing

# Load all images in the folder 'Image folder'
image_list = []
for filename in glob.glob(r'./../Image folder/*.bmp'):
    im = Image.open(filename).convert("LA")
    image_list.append(im)

input_pixels = image_list[0].load()
width, height = image_list[0].width, image_list[0].height
for i in range(len(image_list)):
    print('Image {} has the following information:'.format(i))
    print(image_list[i].format, image_list[i].size, image_list[i].mode)


# Create output image
output_image = Image.new("RGB", input_image.size)
draw = ImageDraw.Draw(output_image)

if np.shape(image_list[0])[2] == 3:
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
    output_image.save(r'./../Image folder/output.bmp')
else
    # Compute convolution between intensity and kernels
    for x in range(1, image_list[0].width - 1):
        for y in range(1, image_list[0].height - 1):
            magx = intensity[x + 1, y] - intensity[x - 1, y]
            magy = intensity[x, y + 1] - intensity[x, y - 1]

            # Draw in black and white the magnitude
            color = int(sqrt(magx ** 2 + magy ** 2))
            draw.point((x, y), (color, color, color))
    output_image.save(r'./../Image folder/output.bmp')
# Similiarty & mean_squared_error
pre_proc = ImagePreProcessing(image_list)
img_ssim, img_error = pre_proc.ssim_and_error(img1=0, img2=1)

print(img_ssim)


