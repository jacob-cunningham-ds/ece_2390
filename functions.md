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
:caption: `imread` syntax
cv2.imread(filename, flags)
:::

|Parameter|Description|
|--|--|
|filename| Name of file to be loaded|
|flags|Takes values of cv2.IMREAD_MODE|

:::{code} python
:caption: `imread` example
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
:caption: `imshow` syntax
plt.imshow(img, cmap, vmin, vmax)
:::

|Paremeter|Description|
|--|--|
|img|The image data array|
|cmap|The colormap; parameter is ignored if img is RGB(A)|
|vmin|Minimum of the colormap|
|vmax|Maximum of the colormap|

:::{code} python
:caption: `imshow` example
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
:footer: [Documentation](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#gaf86c09fe702ed037c03c2bc603ceab14)

Converts an image from one color space to another.

The function converts an input image from one color space to another. In case of a transformation to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.

:::{code} python
:caption: `cvtColor` syntax
cv2.cvtColor(src, dst, code, dstnCn = 0)
:::

|Parameters|Description|
|--|--|
|src| input image|
|dst| output image of the same size and depth as src|
|code| color space conversion code|
|dstCn| number of channles in the destination image|

:::{code} python
:caption: `cvtColor` example
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
:caption: `split` syntax
cv2.split(src)
:::

|Parameters|Description|
|--|--|
|src| input image|

:::{code} python
:caption: `split` example
# split the image into blue, green, and red components
b, g, r = cv2.split(img)
:::
::::

## merge

(card-merge)=
::::{card}
:header: merge
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga7d7b4d6c6ee504b30a20b1680029c7b4)

Creates one multi-channel array out of several single-channel ones.

:::{code} python
:caption: `merge` syntax
cv2.merge(channels)
:::

|Parameters|Description|
|--|--|
|channels| tuple of channels to merge|

:::{code} python
:caption: `merge` example
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
:caption: `imwrite` syntax
cv2.imwrite(filename, img)
:::

|Parameters|Description|
|--|--|
|filename| string; name of output file|
|img| the image to save|

:::{code} python
:caption: `imwrite` example
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
:caption: `resize` syntax
cv2.resize(src, dsize, dst, fx, fy, interpolation)
:::

|Parameters|Description|
|--|--|
|src| input image|
|dsize| output image size|
|fx| scale factor along the horizontal axis|
|fy| scale factor along the vertical axis|
|interpolation| interpolation method|

:::{code} python
:caption: `resize` example

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
:caption: `flip` syntax
cv2.flip(src, flipCode)
:::

|Parameters|Description|
|--|--|
|src| input image|
|flipCode| flag to specify how to flip the image|


:::{code} python
:caption: `flip` example
# flip imagine about x-axis
img_flipped_vert = cv2.flip(img, 0)

# flip image about y-axis
img_flipped_horiz = cv2.flip(img, 1)

# flip image about both axes
img_flipped_both = cv2.flip(img, -1)
:::
::::

## line

(card-line)=
::::{card}
:header: line
:footer: [Documentation](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2)

Draws a line segment connecting two points.

The function line draws the line segment between pt1 and pt2 points in the image. The line is clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased lines are drawn using Gaussian filtering.

:::{code} python
:caption: `line` syntax
cv2.line(img, pt1, pt2, color, thickness, lineType)
:::

|Parameters|Description|
|--|--|
|img| Image on which we will draw the line|
|pt1| First point (x, y) of the line segment|
|pt2| Second point (x, y) of the line segment|
|color| Color of the line which will be drawn|
|thickness| Integer specifying the thickness of the line; default value is 1|
|lineType| Type of line. Default is 8-connected line. Usually `cv2.LINE_AA` is used|

:::{code} pyton
:caption: `line` example
start = (200, 100)
stop = (400, 100)
yellow = (0, 255, 255)

cv2.line(imageLine, start, stop, yellow, thickness=5, lineType=cv2.LINE_AA)
:::
::::


