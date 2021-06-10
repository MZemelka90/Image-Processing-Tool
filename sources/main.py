import numpy as np
from PIL import Image, ImageDraw
from skimage import data, img_as_float
# load image
file_name = 'ID0101_0001.bmp'
input_image = Image.open(r'./../Image folder/' + file_name)
input_pixels = input_image.load()
width, height = input_image.width, input_image.height

# Create output image
output_image = Image.new("RGB", input_image.size)
draw = ImageDraw.Draw(output_image)

# convert to grayscale
intensity = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        intensity[x, y] = sum(input_pixels[x, y]) / 3

# Compute convolution between intensity and kernels
for x in range(1, input_image.width -1):
    for y in range(1, input_image.height - 1):
        magx = intensity[x+1,y] - intensity [x-1, y]
        magy = intensity[x, y + 1] - intensity[x, y - 1]

        # Draw in black and white the magnitude
        color = int(sqrt(magx**2 + magy**2))
        draw.point((x, y), (color, color, color))
output_image.save("output.png")



