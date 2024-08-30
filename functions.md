---
title: Function Reference
---

## imread

(card-imread)=
::::{card}
:header: imread
:footer: [Documentation](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html)

Loads an image from a file.

The function imread loads an image from the specified file and returns it. If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format), the function returns an empty matrix.

:::{code} python
:caption: `imread` Example
# read in a BGR image
img = cv2.imread(os.path.relpath('assets/example.jpg'), cv2.IMREAD_COLOR)

# read in a grayscale image
img = cv2.imread(os.path.replath('assets/example.jpg'), cv2.IMREAD_GRAYSCALE)
:::
::::

## imshow

(card-imshow)=
::::{card}
:header: imshow
:footer: [Documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)

Display data as an image, i.e., on a 2D regular raster.

The input may either be actual RGB(A) data, or 2D scalar data, which will be rendered as a pseudocolor image. For displaying a grayscale image, set up the colormapping using the parameters `cmap='gray', vmin=0, vmax=255`.

:::{code} python
:caption: `imshow` Example
# create a subplot of a RGB(A) image and grayscale image
plt.figure()
plt.subplot(121); plt.imshow(img); plt.title('RGB(A) Image')
plt.subplot(122); plt.imshow(img, cmap="gray", vmin=0, vmax=255); plt.title('Grayscale Image')
plt.show()
:::
::::

## cvtColor

(card-cvtColor)=
::::{card}
:header: cvtColor
:footer: [Documentation](https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab)

Converts an image from one color space to another.

The function converts an input image from one color space to another. In case of a transformation to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.

:::{code} python
:caption: `cvtColor` Example
# convert BGR to RGB
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert BGR to Gray
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
:::
::::

## split

(card-split)=
::::{card}
:header: split
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga0547c7fed86152d7e9d0096029c8518a)

Divides a multi-channel array into several single-channel arrays.

:::{code} python
:caption: `split` Example
# split the image into blue, green, and red components
b, g, r = cv2.split(img)
:::
::::

## merge

(card-merge)=
::::{card}
:header: merge
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga61f2f2bde4a0a0154b2333ea504fab1d)

Creates one multi-channel array out of several single-channel ones.

:::{code} python
:caption: `merge` Example
# split the image into blue, green, and red components
b, g, r = cv.split(img)

# alter the blue channel
b = b * 0 

# merge the image back together
new_img = cv2.merge((b, g, r))
:::
::::

## imwrite

(card-imwrite)=
::::{card}
:header: imwrite
:footer: [Documentation](https://docs.opencv.org/4.5.1/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce)

Saves an image to a specified file.

The function imwrite saves the image to the specified file. The image format is chosen based on the filename extension (see cv::imread for the list of extensions). In general, only 8-bit single-channel or 3-channel (with 'BGR' channel order) images can be saved using this function.

:::{code} python
:caption: `imwrite` Example
# save the image
cv2.imwrite('output.png', img);
:::
::::

## resize

(card-resize)=
::::{card}
:header: resize
:footer: [Documentation](https://docs.opencv.org/4.5.0/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)

Resizes an image.

The function resize resizes the image `src` down to or up to the specified size. Note that the initial `dst` type or size are not taken into account. Instead, the size and type are derived from the `src`, `dsize`, `fx`, and `fy`.

:::{code} python
:caption: `resize` Example

# specifying a scaling factor
img_2X = cv2.resize(img, dsize=None, fx=2, fy=2)
plt.imshow(img_2X)
plt.show()

# specifying the exact output size
img_specific = cv2.(img, dsize=(100, 200))
plt.imshow(img_specific)
plt.show()
:::
::::

## flip

(card-flip)=
::::{card}
:header: flip
:footer: [Documentation](https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441)

Flips a 2D array around vertical, horizontal, or both axes. 

:::{code} python
:caption: `flip` Example
# flip imagine about x-axis
img_flipped_vert = cv2.flip(img, 0)

# flip image about y-axis
img_flipped_horiz = cv2.flip(img, 1)

# flip image about both axes
img_flipped_both = cv2.flip(img, -1)
:::
::::




