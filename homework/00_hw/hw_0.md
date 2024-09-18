---
title: Homework 0
subtitle: Getting started with OpenCV
downloads:
  - file: hw_0.ipynb
    title: hw_0.ipynb
---

## Assignment
The purpose of this assignment is to load in an image, perform operations on the image, and save the image.

## Import modules
:::{code} python
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
:::

## Load the image
:::{code} python
# read in the image
img = cv2.imread(os.path.relpath('Data/apollo_11_launch.jpg'), cv2.IMREAD_GRAYSCALE)
:::

## View the image
:::{code} python
# render the image
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray', vmin=0, vmax=255)
ax.set_title('Apollo 11 Launch')
plt.show()
:::

:::{figure} Data/apollo_11_launch_loaded.jpg
:align: center
:::

## Crop the image
:::{code} python
# take rows 100 through 700 and columns 400 through 800
img_cropped = img[100:701, 400:801]

# render the image
fig, ax = plt.subplots()
ax.imshow(img_cropped, cmap='gray', vmin=0, vmax=255)
ax.set_title('Apollo 11 Launch Cropped')
plt.show()
:::

:::{figure} Data/apollo_11_launch_cropped.jpg
:align: center
:::

## Edit the brightness
:::{code} python
# create a matrix of ones with the same dimensions of the image
matrix = np.ones(img.shape, dtype="uint8")
matrix *= 50 # scale by 50

# brighten the image by adding the matrix to the image
img_bright = cv2.add(img, matrix)

# render the image
fig, ax = plt.subplots()
ax.imshow(img_bright, cmap='gray', vmin=0, vmax=255)
ax.set_title('Apollo 11 Launch Brighter')
plt.show()
:::

:::{figure} Data/apollo_11_launch_brighter.jpg
:align: center
:::

## Rotate the image
:::{code} python
# flip the image on the horizontal axis
img_flipped = cv2.flip(img, 0)

# render the image
fig, ax = plt.subplots()
ax.imshow(img_flipped, cmap='gray', vmin=0, vmax=255)
ax.set_title('Apollo 11 Launch Rotated')
plt.show()
:::

:::{figure} Data/apollo_11_launch_rotated.jpg
:align: center
:::

## Save the image
:::{code} python
cv2.imwrite(os.path.relpath('Data/apollo_11_mod.jpg'), img_flipped)
:::