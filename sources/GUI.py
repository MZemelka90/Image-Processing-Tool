import numpy as np
from PIL import ImageDraw
import io
import PIL.Image
from math import sqrt, pi, cos, sin
from canny import canny_edge_detector
from collections import defaultdict
import glob
import cv2
import matplotlib.pyplot as plt
import os
from tkinter import *
from tkinter import ttk
from ttkthemes import ThemedTk
import tkinter as tk
import tkinter.filedialog
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("TkAgg")

Image_Processing = ThemedTk(theme="arc")
Image_Processing.state('zoomed')
Image_Processing.title("Image Processing Tool")
Image_Processing.geometry('1200x1600')

# Creating Menubar
menubar = Menu(Image_Processing)

# Adding File Menu and commands
file = Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=file)
file.add_command(label='New File', command=None)
file.add_command(label='Open...', command=lambda: open_file())
file.add_command(label='Save', command=lambda: save_diff_matrix())
file.add_separator()
file.add_command(label='Exit', command=Image_Processing.destroy)

# Adding Edit Menu and commands
edit = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Edit', menu=edit)
edit.add_command(label='Cut', command=None)
edit.add_command(label='Copy', command=None)
edit.add_command(label='Paste', command=None)
edit.add_command(label='Select All', command=None)
edit.add_separator()
edit.add_command(label='Find...', command=None)
edit.add_command(label='Find again', command=None)

# Adding Help Menu
help_ = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Help', menu=help_)
help_.add_command(label='Tk Help', command=None)
help_.add_command(label='Demo', command=None)
help_.add_separator()
help_.add_command(label='About Tk', command=None)


def open_file():
    global files, file_name, im_array, image_list
    file_name = tkinter.filedialog.askopenfilenames(filetypes=(("All files", "*"), ("Template files", "*.type")))
    filepath, filetype, files = [], [], []
    for i in range(0, len(file_name)):
        filepath.append(os.path.splitext(file_name[i])[0])
        filetype.append(os.path.splitext(file_name[i])[1])
        files.append(os.path.basename(file_name[i]))

    # Sort & prepare Images
    # --------------------------------------------------------------------------------------------------------------------
    # Populate the variables option Menu
    opt.config(value=files)

    image_list, im_array = [], []
    for i in range(0, len(file_name)):
        im = PIL.Image.open(file_name[i])
        image_list.append(im)

    # Make arrays from the BMP information
    for i in range(len(image_list)):
        im_array.append(np.array(image_list[i], dtype='int16'))

    return filetype, files, im_array, image_list


def show_image():

    ax1.clear()
    ax1.imshow(PIL.Image.open('./../Image folder/' + opt.get()), cmap='gray')
    figure.canvas.draw()


def difference_matrix():
    global diff_matrix
    img_1 = int(im_1.get())
    img_2 = int(im_2.get())
    diff_matrix = np.zeros_like(im_array[img_1])
    for i in range(0, np.shape(im_array)[1]):
        for j in range(0, np.shape(im_array)[2]):
            diff_matrix[i][j] = abs(im_array[img_1][i][j] - im_array[img_2][i][j])
            if 100 > diff_matrix[i][j]:
                diff_matrix[i][j] = 20
            if diff_matrix[i][j] > 180:
                diff_matrix[i][j] = 250
    print(np.min(diff_matrix), np.max(diff_matrix))
    print(diff_matrix)
    ax2.clear()
    ax2.imshow(diff_matrix, cmap='gray', vmin=0, vmax=255)
    figure_2.canvas.draw()
    return diff_matrix


def save_diff_matrix():
    file_types = [('All Files', '*.*'),
             ('Image', '*.png'),
             ('Bitmap', '*.bmp'),
             ('JPEG', '*.jpeg')
                ]
    output_file = tkinter.filedialog.asksaveasfile(filetypes=file_types)
    output_file = output_file.name
    print(np.shape(output_file))

    ax2.imshow(diff_matrix, cmap='gray', vmin=0, vmax=255)
    figure_2.savefig(output_file)


def detect_circles(param1, param2, minRad, maxRad):
    p1 = int(param1.get())
    p2 = int(param2.get())
    minR = int(minRad.get())
    maxR = int(maxRad.get())

    # Read image.
    img = cv2.imread(r'./../Image folder/' + opt.get(), cv2.IMREAD_GRAYSCALE)
    print(img)
    print(img.dtype)
    print('Shape of the image is {}'.format(np.shape(img)))
    print('Min. {}, Max. {}'.format(np.min(img), np.max(img)))

    # Apply Hough transform on the image.
    detected_circles = cv2.HoughCircles(img,
                                        cv2.HOUGH_GRADIENT, 1, 45, param1=p1,
                                        param2=p2, minRadius=minR, maxRadius=maxR)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", img)
            cv2.waitKey(0)

# display Menu
Image_Processing.config(menu=menubar)

window1 = LabelFrame(Image_Processing, text='Image Processing', font=("Arial Bold", 12))
window1.grid(row=0, column=0)


button_2 = Button(window1, text="Image", height=2, width=10, command=lambda: show_image())
button_2.grid(row=0, column=0)
button_3 = Button(window1, text="Difference", height=2, width=10, command=lambda: difference_matrix())
button_3.grid(row=4, column=0)
button_4 = Button(window1, text="Circle Detection", height=2, width=10, command=lambda: detect_circles(param1, param2, minRad, maxRad))
button_4.grid(row=4, column=1)

dummy = []
opt = ttk.Combobox(window1, value=dummy, state='readonly')
opt.config(width=15)
opt.grid(row=1, column=1)
Label(window1, text='Loaded files: ', anchor=N).grid(row=1, column=0)

im_1 = Entry(window1)
im_1.grid(row=2, column=1)
im_1.insert(0, 0)
Label(window1, text='Image #1', anchor=W).grid(row=2, column=0)

im_2 = Entry(window1)
im_2.grid(row=3, column=1)
im_2.insert(0, 1)
Label(window1, text='Image #2', anchor=W).grid(row=3, column=0)

pxpmmx = Entry(window1)
pxpmmx.grid(row=5, column=1)
pxpmmx.insert(0, 71)
Label(window1, text='Pixel / mm -x', anchor=W).grid(row=5, column=0)

pxpmmy = Entry(window1)
pxpmmy.grid(row=6, column=1)
pxpmmy.insert(0, 71)
Label(window1, text='Pixel / mm -y', anchor=W).grid(row=6, column=0)

FPS = Entry(window1)
FPS.grid(row=7, column=1)
FPS.insert(0, 1000)
Label(window1, text='Frames per second', anchor=W).grid(row=7, column=0)

# PlottingFrames
windowPlot_1 = LabelFrame(Image_Processing, text='Plots Preprocessing', font=("Arial Bold", 12))
windowPlot_1.grid(column=5, row=0, padx=10, pady=0)

figure = plt.Figure(figsize=(6, 8), dpi=122)
figure.tight_layout()
ax1 = figure.add_subplot(111)
chart_type = FigureCanvasTkAgg(figure, windowPlot_1)
chart_type.get_tk_widget().grid(padx=50, pady=0, columnspan=2)

windowPlot_2 = LabelFrame(Image_Processing, text='Plots Postprocessing', font=("Arial Bold", 12))
windowPlot_2.grid(column=7, row=0, padx=10, pady=0)

figure_2 = plt.Figure(figsize=(6, 8), dpi=122)
figure_2.tight_layout()
ax2 = figure_2.add_subplot(111)
chart_type = FigureCanvasTkAgg(figure_2, windowPlot_2)
chart_type.get_tk_widget().grid(padx=50, pady=0, columnspan=2)


param1 = Entry(window1)
param1.grid(row=8, column=1)
param1.insert(0, 40)
Label(window1, text='Param #1', anchor=W).grid(row=8, column=0)

param2 = Entry(window1)
param2.grid(row=9, column=1)
param2.insert(0, 30)
Label(window1, text='Param #2', anchor=W).grid(row=9, column=0)

minRad = Entry(window1)
minRad.grid(row=10, column=1)
minRad.insert(0, 1)
Label(window1, text='min circle Radius', anchor=W).grid(row=10, column=0)

maxRad = Entry(window1)
maxRad.grid(row=11, column=1)
maxRad.insert(0, 50)
Label(window1, text='max circle Radius', anchor=W).grid(row=11, column=0)

Image_Processing.mainloop()
