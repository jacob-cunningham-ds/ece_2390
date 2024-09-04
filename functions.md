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
yellow = (0, 255, 255) # in BGR format

cv2.line(imageLine, start, stop, yellow, thickness=5, lineType=cv2.LINE_AA)
:::
::::

## circle

(card-circle)=
::::{card}
:header: circle
:footer: [Documentation](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670)

Draws a circle.

:::{code} python
:caption: `circle` syntax
cv2.circle(img, center, radius, color, thickness, lineType)
:::

|Parameters|Description|
|--|--|
|img| Image on which we will draw the circle|
|center| center of the circle|
|radius| radius of the circle|
|color| Color of the circle which will be drawn|
|thickness| Integer specifying the thickness of the circle; default value is 1|
|lineType| Type of line. Default is 8-connected line. Usually `cv2.LINE_AA` is used|

:::{code} python
:caption: `circle` example
cv2.circle(img, (900, 500), 100, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
:::
::::

## rectangle

(card-rectangle)=
::::{card}
:header: rectangle
:footer: [Documentation](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9)

Draws a simple, thick, or filled up-right rectangle. 

:::{code} python
:caption: `rectangle` syntax
cv2.rectangle(img, pt1, pt2, color, thickness, lineType)
:::

|Parameters|Description|
|--|--|
|img| Image on which we will draw the rectangle|
|pt1| vertex of the rectangle|
|pt2| vertex of the rectangle opposite of pt1|
|color| Color of the rectangle which will be drawn|
|thickness| Integer specifying the thickness of the rectangle; default value is 1|
|lineType| Type of line. Default is 8-connected line. Usually `cv2.LINE_AA` is used|

:::{code} python
:caption: `rectangle` example
cv2.rectangle(img, (500, 100), (700, 600), (255, 0, 255), thickness=5, lineType=cv2.LINE_AA)
:::
::::

## putText

(card-putText)=
::::{card}
:header: putText
:footer: [Documentation](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576)

Draws a text string.

:::{code} python
:caption: `putText` syntax
cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)
:::

|Parameters|Description|
|--|--|
|img| Image on which we will draw the text|
|text| text string to be drawn|
|org| bottom left corner of text|
|fontFace| font type|
|fontScale| size of font|
|color| Color of the text which will be drawn|
|thickness| Integer specifying the thickness of the text; default value is 1|
|lineType| Type of line. Default is 8-connected line. Usually `cv2.LINE_AA` is used|

:::{code} python
:caption: `putText` example
cv2.putText(img, 'Hello World!', (200, 700), cv2.FONT_HERSHEY_PLAIN, 2.3, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
:::
::::

## add

(card-add)=
::::{card}
:header: add
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6)

Calculates the per-element sum of two arrays or an array and a scalar. 

:::{code} python
:caption: `add` syntax
cv2.add(arr1, arr2)
:::

|Parameters|Description|
|--|--|
|arr1| First array to add|
|arr2| Second array to add|

:::{code} python
:caption: `add` example
# create an array the same size of the image
mat_add = np.ones(img.shape, dtype="uint8") * 50

cv2.add(img, mat_add)
:::
::::

## subtract

(card-subtract)=
::::{card}
:header: subtract
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gaa0f00d98b4b5edeaeb7b8333b2de353b)

Calculates the per-element difference of two arrays or an array and a scalar. 

:::{code} python
:caption: `subtract` syntax
cv2.subtract(arr1, arr2)
:::

|Parameters|Description|
|--|--|
|arr1| First array to subtract|
|arr2| Second array to subtract|

:::{code} python
:caption: `subtract` example
# create an array the same size of the image
mat_sub = np.ones(img.shape, dtype="uint8") * 50

cv2.subtract(img, mat_sub)
:::
::::

## multiply

(card-multiply)=
::::{card}
:header: multiply
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga979d898a58d7f61c53003e162e7ad89f)

Calculates the per-element scaled product of two arrays.

:::{code} python
:caption: `multiply` syntax
cv2.multiply(arr1, arr2)
:::

|Parameters|Description|
|--|--|
|arr1| First array to multiply|
|arr2| Second array to multiply|

:::{code} python
:caption: `subtract` example
# create an array the same size of the image
mat_mult = np.ones(img.shape, dtype="uint8") * 50

cv2.multiply(img, mat_mult)
:::
::::

## clip

(card-clip)=
::::{card}
:header: clip
:footer: [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.clip.html)

Clip (limit) the values in an array.

Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.

:::{code} python
:caption: `clip` syntax
np.clip(a, amin, amax, out)
:::

|Parameters|Description|
|--|--|
|a| array to clip|
|amin| lower bound clip value|
|amax| upper bound clip value|
|out| output to new array|

:::{code} python
:caption: `clip` example
np.clip(img, 0, 255)
:::
::::

## threshold

(card-threshold)=
::::{card}
:header: threshold
:footer: [Documentation](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)

Applies a fixed-level threshold to each array element.

The function applies fixed-level thresholding to a multiple-channel array. The function is typically used to get a bi-level (binary) image out of a grayscale image ( compare could be also used for this purpose) or for removing a noise, that is, filtering out pixels with too small or too large values. There are several types of thresholding supported by the function. They are determined by type parameter.

:::{code} python
:caption: `threshold` syntax
retval, dst = cv2.threshold(src, thresh, maxval, type)
:::

|Parameters|Description|
|--|--|
|src| the source image (grayscale)|
|thresh| The threshold value|
|maxval| The maximum value that will be assigned to the pixels exceeding the threshold|
|type| The type of thresholding to apply|
|retval|The threshold value used|
|dst|The output image after thresholding|

:::{code} python
:caption: `threshold` example
retval, img_thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
:::
::::

## adaptiveThreshold

(card-adaptiveThreshold)=
::::{card}
:header: adaptiveThreshold
:footer: [Documentation](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)

Applies an adaptive threshold to an array.

The function transforms a grayscale image to a binary image according to the formulae:

__THRESH_BINARY__

$$
dst(x,y) = 
\begin{cases}
  maxValue & \text{if } src(x,y) > T(x,y) \\
  0 & \text{otherwise}
\end{cases}
$$

__THRESH_BINARY_INV__

$$
dst(x,y) = 
\begin{cases}
  0 & \text{if } src(x,y) > T(x,y) \\
  maxValue & \text{otherwise}
\end{cases}
$$

where $T(x,y)$ is a threshold value calculated individually for each pixel.

:::{code} python
dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
:::

|Parameters|Description|
|--|--|
|src| the source image (grayscale)|
|maxValue| The maximum value that will be assigned to the pixels exceeding the threshold|
|adaptiveMethod| The method used to calculate the threshold value for the local region|
|thresholdType| The type of thresholding to apply|
|blockSize|The size of the neighborhood area (a square) used to calculate the threshold for each pixel.|
|C|A constant that is subtracted from the computed mean or weighted mean to fine-tune the thresholding|

:::{code} python
:caption: `adaptiveThreshold` example
dst = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11, 7)
:::
::::

## bitwise_and

(card-bitwise_and)=
(card-bitwise_and)=
::::{card}
:header: bitwise_and
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14)

Performs a bitwise AND operation on two arrays (images).

The function computes the bitwise conjunction of the two input arrays (images) element-wise, with an optional mask. The output array has the same size and type as the input arrays. This operation is often used in masking operations, where the region of interest in an image is extracted using a binary mask.

:::{code} python
:caption: `bitwise_and` syntax
dst = cv2.bitwise_and(src1, src2, mask=None)
:::

|Parameters|Description|
|--|--|
|src1| First input array (image)|
|src2| Second input array (image)|
|mask| Optional mask array. If the mask is not empty, the operation is applied only to the pixels where the mask is non-zero|
|dst| Output array (image) that has the same size and type as the input arrays|

:::{code} python
:caption: `bitwise_and` example
result = cv2.bitwise_and(image1, image2)
:::
::::

## bitwise_or

(card-bitwise_or)=
::::{card}
:header: bitwise_or
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gab85523db362a4e26ff0c703793a719b4)

Performs a bitwise OR operation on two arrays (images).

The function computes the bitwise disjunction of the two input arrays (images) element-wise, with an optional mask. The output array has the same size and type as the input arrays. This operation is commonly used to combine regions of interest from two images.

:::{code} python
:caption: `bitwise_or` syntax
dst = cv2.bitwise_or(src1, src2, mask=None)
:::

|Parameters|Description|
|--|--|
|src1| First input array (image)|
|src2| Second input array (image)|
|mask| Optional mask array. If the mask is not empty, the operation is applied only to the pixels where the mask is non-zero|
|dst| Output array (image) that has the same size and type as the input arrays|

:::{code} python
:caption: `bitwise_or` example
result = cv2.bitwise_or(image1, image2)
:::
::::

## bitwise_xor

(card-bitwise_xor)=
::::{card}
:header: bitwise_xor
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga84b2d8188ce506593dcc3f8cd00e8e2c)

Performs a bitwise XOR operation on two arrays (images).

The function computes the bitwise exclusive OR of the two input arrays (images) element-wise, with an optional mask. The output array has the same size and type as the input arrays. This operation is useful for highlighting differences between two images.

:::{code} python
:caption: `bitwise_xor` syntax
dst = cv2.bitwise_xor(src1, src2, mask=None)
:::

|Parameters|Description|
|--|--|
|src1| First input array (image)|
|src2| Second input array (image)|
|mask| Optional mask array. If the mask is not empty, the operation is applied only to the pixels where the mask is non-zero|
|dst| Output array (image) that has the same size and type as the input arrays|

:::{code} python
:caption: `bitwise_xor` example
result = cv2.bitwise_xor(image1, image2)
:::
::::

## bitwise_not

(card-bitwise_not)=
::::{card}
:header: bitwise_not
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga0002cf8b418479f4cb49a75442baee2f)

Performs a bitwise NOT operation on an array (image).

The function inverts every bit of an array element-wise. The output image will have its pixel values inverted, such that all 0 bits are set to 1 and all 1 bits are set to 0. This operation is commonly used to create a negative of an image.

:::{code} python
:caption: `bitwise_not` syntax
dst = cv2.bitwise_not(src)
:::

|Parameters|Description|
|--|--|
|src| Input array (image)|
|dst| Output array (image) that has the same size and type as the input array|

:::{code} python
:caption: `bitwise_not` example
result = cv2.bitwise_not(image)
:::
::::

## erode

(card-erode)=
::::{card}
:header: erode
:footer: [Documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb)

Performs an erosion operation on an image.

The `cv2.erode` function applies erosion to an image, which reduces the boundaries of the foreground objects. Erosion is achieved by moving a structuring element (kernel) across the image and shrinking the objects based on the shape and size of the kernel. This operation is useful for removing small noise, separating objects that are close together, and reducing the size of objects.

:::{code} python
:caption: `erode` syntax
dst = cv2.erode(src, kernel, iterations=1)
:::

|Parameters|Description|
|--|--|
|src| The source image on which erosion is to be applied|
|kernel| Structuring element used for erosion, typically created using `np.ones` or `cv2.getStructuringElement`|
|iterations| Number of times erosion is applied. Defaults to 1|
|dst| Output image after erosion|

:::{code} python
:caption: `erode` example
kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(img, kernel, iterations=1)
:::
::::

## dilate

(card-dilate)=
::::{card}
:header: dilate
:footer: [Documentation](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c)

Expands the boundaries of objects in a binary image.

The `cv2.dilate` function applies a dilation operation to an image, which increases the boundaries of the foreground objects. Dilation is performed by placing a structuring element (kernel) over the image and expanding the white (foreground) regions based on the shape and size of the kernel. This operation is useful for filling small holes, connecting disjointed components, or increasing the size of objects in the image.

:::{code} python
:caption: `dilate` syntax
dst = cv2.dilate(src, kernel, iterations=1)
:::

|Parameters|Description|
|--|--|
|src| The source image on which dilation is to be applied|
|kernel| Structuring element used for dilation, typically created using `np.ones` or `cv2.getStructuringElement`|
|iterations| Number of times dilation is applied. Defaults to 1|
|dst| Output image after dilation|

:::{code} python
:caption: `dilate` example
kernel = np.ones((5, 5), np.uint8)
dilated_image = cv2.dilate(img, kernel, iterations=1)
:::
::::

## morphologyEx

(card-morphologyEx)=
::::{card}
:header: morphologyEx
:footer: [Documentation](https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html)

Applies advanced morphological transformations to an image.

The `cv2.morphologyEx` function performs various morphological operations, such as opening, closing, gradient, top hat, and black hat, on an image using a specified structuring element (kernel). These operations are combinations of basic erosions and dilations, used to refine image regions, remove noise, or detect specific shapes.

:::{code} python
:caption: `morphologyEx` syntax
dst = cv2.morphologyEx(src, op, kernel, iterations=1)
:::

|Parameters|Description|
|--|--|
|src| The source image on which the morphological operation is to be applied|
|op| Type of morphological operation to perform (e.g., `cv2.MORPH_OPEN`, `cv2.MORPH_CLOSE`, `cv2.MORPH_GRADIENT`, etc.)|
|kernel| Structuring element used for the operation, typically created using `np.ones` or `cv2.getStructuringElement`|
|iterations| Number of times the operation is applied. Defaults to 1|
|dst| Output image after applying the morphological operation|

:::{code} python
:caption: `morphologyEx` example
kernel = np.ones((5, 5), np.uint8)
opened_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
:::
::::

## colorbar

(card-colorbar)=
::::{card}
:header: colorbar
:footer: [Documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html)

Adds a colorbar to a plot, representing the mapping of data values to colors.

The `plt.colorbar` function in Matplotlib creates a colorbar that corresponds to the colormap and intensity values of the image or plot. The colorbar provides a visual representation of the mapping between data values and colors used in the plot, which is particularly useful for interpreting the meaning of colors in an image or heatmap.

:::{code} python
:caption: `colorbar` syntax
plt.colorbar(mappable=None, cax=None, ax=None, **kwargs)
:::

|Parameters|Description|
|--|--|
|mappable| The image object (e.g., returned by `plt.imshow`) to which the colorbar applies.|
|cax| An optional axes into which the colorbar will be drawn.|
|ax| An optional axes onto which the colorbar will be attached.|
|**kwargs| Additional optional keyword arguments to customize the colorbar.|
|Return| The colorbar object that has been added to the plot.|

:::{code} python
:caption: `colorbar` example
img_display = plt.imshow(image_data, cmap='viridis')
plt.colorbar(img_display)
plt.show()
:::
::::









































