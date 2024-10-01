---
title: Function Reference
---

# OpenCV

## adaptiveThreshold

(card-adaptiveThreshold)=
::::{card}
:header: adaptiveThreshold
:footer: [Documentation](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)

Applies an adaptive threshold to an image.

The `cv2.adaptiveThreshold` function converts a grayscale image to a binary image by calculating a threshold for small regions of the image. This allows different thresholds for different parts of the image, which is useful when the image lighting varies across the scene.

:::{code} python
:caption: `adaptiveThreshold` syntax
dst = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
:::

|Parameters|Description|
|--|--|
|src| Input grayscale image|
|maxValue| Maximum value to assign to the pixels that meet the threshold condition|
|adaptiveMethod| Method used to calculate the threshold (e.g., `cv2.ADAPTIVE_THRESH_MEAN_C` or `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`)|
|thresholdType| Type of thresholding (`cv2.THRESH_BINARY` or `cv2.THRESH_BINARY_INV`)|
|blockSize| Size of the neighborhood area used to calculate the threshold for each pixel (must be an odd number)|
|C| Constant subtracted from the mean or weighted mean (fine-tuning parameter)|

:::{code} python
:caption: `adaptiveThreshold` example
# Apply adaptive mean thresholding
dst = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply adaptive Gaussian thresholding
dst = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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

## bitwise_and

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

## cv2.blur

(card-blur)=
::::{card}
:header: cv2.blur
:footer: [Documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37)

Applies a simple average (mean) filter to an image. This filter smooths the image by averaging pixel values in a specified kernel window.

:::{code} python
:caption: `blur` syntax
dst = cv2.blur(src, ksize[, anchor[, borderType]])
:::

|Parameters|Description|
|--|--|
|src| Input image (can be grayscale or color)|
|ksize| Size of the kernel (e.g., (5, 5) for a 5x5 kernel)|
|anchor| Anchor point within the kernel (default is (-1,-1), which indicates the kernel center)|
|borderType| Pixel extrapolation method (default is `cv2.BORDER_DEFAULT`)|

### Border Types:
- `cv2.BORDER_CONSTANT`: Pads with a constant value (set with `value`)
- `cv2.BORDER_REPLICATE`: Repeats the border pixels
- `cv2.BORDER_REFLECT`: Reflects the border elements
- `cv2.BORDER_WRAP`: Wraps around the image (used less often)

### Example:

:::{code} python
:caption: `blur` example
import cv2

# Read the image
image = cv2.imread('image.jpg')

# Apply a 5x5 mean filter
blurred_image = cv2.blur(image, (5, 5))

# Display the result
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
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

## cv2.convertScaleAbs

(card-convertScaleAbs)=
::::{card}
:header: cv2.convertScaleAbs
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga4f8b7a518930d411619bb15017f3f0ff)

`cv2.convertScaleAbs` is used to scale, calculate the absolute values, and convert the result to 8-bit type (`CV_8U`). It is commonly used after applying image filters or transformations (like `cv2.Sobel`) that may produce floating-point output, which needs to be converted to 8-bit for visualization.

:::{code} python
:caption: `convertScaleAbs` syntax
dst = cv2.convertScaleAbs(src, alpha=1, beta=0)
:::

|Parameters|Description|
|--|--|
|src| Input image (can be of any type, typically float32 or float64 for scaling and conversion)|
|alpha| Optional scale factor (default is 1)|
|beta| Optional added value (default is 0)|

### Example:

:::{code} python
:caption: `convertScaleAbs` example
import cv2
import numpy as np

# Read the image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Sobel filter to detect edges (float output)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# Convert the result to 8-bit using convertScaleAbs
sobel_x_8bit = cv2.convertScaleAbs(sobel_x)

# Display the result
cv2.imshow('Sobel X 8-bit', sobel_x_8bit)
cv2.waitKey(0)
cv2.destroyAllWindows()
:::
::::

## cvtColor

(card-cvtColor)=
::::{card}
:header: cvtColor
:footer: [Documentation](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#gaf86c09fe702ed037c03c2bc603ceab14)

Converts an image from one color space to another.

The function converts an input image from one color space to another. In case of a transformation to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the bytes are reversed).

:::{code} python
:caption: `cvtColor` syntax
cv2.cvtColor(src, dst, code, dstnCn = 0)
:::

|Parameters|Description|
|--|--|
|src| input image|
|dst| output image of the same size and depth as src|
|code| color space conversion code|
|dstCn| number of channels in the destination image|

:::{code} python
:caption: `cvtColor` example
# convert BGR to RGB
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert BGR to Gray
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

## equalizeHist

(card-equalizeHist)=
::::{card}
:header: equalizeHist
:footer: [Documentation](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)

Equalizes the histogram of a grayscale image.

The `cv2.equalizeHist` function improves the contrast of an image by spreading out the most frequent intensity values. This operation is particularly useful for images that have poor contrast due to a narrow range of intensity values. It works only on single-channel images (grayscale).

:::{code} python
:caption: `equalizeHist` syntax
dst = cv2.equalizeHist(src)
:::

|Parameters|Description|
|--|--|
|src| Input grayscale image|
|dst| Output image after histogram equalization|

:::{code} python
:caption: `equalizeHist` example
# Apply histogram equalization to a grayscale image
equalized_img = cv2.equalizeHist(gray_img)

# Display the original and equalized images
plt.subplot(121); plt.imshow(gray_img, cmap='gray'); plt.title('Original')
plt.subplot(122); plt.imshow(equalized_img, cmap='gray'); plt.title('Equalized')
plt.show()
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

## cv2.filter2D

(card-filter2D)=
::::{card}
:header: cv2.filter2D
:footer: [Documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga3cc062f1c14137a0e776e941d1bdaa6d)

`cv2.filter2D` is used to apply a custom linear filter to an image. It convolves the image with a specified kernel, allowing you to create custom filters for operations like sharpening, blurring, edge detection, and more.

:::{code} python
:caption: `filter2D` syntax
dst = cv2.filter2D(src, ddepth, kernel, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
:::

|Parameters|Description|
|--|--|
|src| Input image (can be of any type)|
|ddepth| Desired depth of the destination image (e.g., `cv2.CV_8U`, `cv2.CV_32F`, etc.)|
|kernel| Convolution kernel (must be a single-channel floating-point matrix)|
|anchor| Anchor point of the kernel (default is `(-1, -1)`, which means the anchor is at the kernel center)|
|delta| Optional value added to the result (default is 0)|
|borderType| Pixel extrapolation method (default is `cv2.BORDER_DEFAULT`)|

### Example:

:::{code} python
:caption: `filter2D` example
import cv2
import numpy as np

# Read the image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Define a custom kernel (e.g., sharpening kernel)
kernel = np.array([[0, -1, 0], 
                   [-1, 5, -1], 
                   [0, -1, 0]], np.float32)

# Apply filter2D to the image
sharpened_img = cv2.filter2D(img, -1, kernel)

# Display the result
cv2.imshow('Sharpened Image', sharpened_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
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

## cv2.inpaint

(card-inpaint)=
::::{card}
:header: cv2.inpaint
:footer: [Documentation](https://docs.opencv.org/4.x/df/d3d/group__photo__inpaint.html)

`cv2.inpaint` is used to restore parts of an image that are obscured by artifacts, noise, or other unwanted features by "inpainting" those areas. You provide a mask that marks the areas to be inpainted, and the algorithm fills in the masked regions using information from the surrounding pixels.

:::{code} python
:caption: `inpaint` syntax
dst = cv2.inpaint(src, inpaintMask, inpaintRadius, flags)
:::

|Parameters|Description|
|--|--|
|src| Input image (can be grayscale or color)|
|inpaintMask| 8-bit, single-channel image, where non-zero pixels indicate the region to be inpainted|
|inpaintRadius| Radius of a circular neighborhood of each point inpainted that is considered for the algorithm|
|flags| Inpainting algorithm to use, can be either `cv2.INPAINT_TELEA` or `cv2.INPAINT_NS`|

### Inpainting Algorithms:
- `cv2.INPAINT_TELEA`: Fast Marching Method-based algorithm (TELEA)
- `cv2.INPAINT_NS`: Navier-Stokes based method for inpainting

### Example:

:::{code} python
:caption: `inpaint` example
import cv2

# Read the image
image = cv2.imread('damaged_image.jpg')

# Create a mask where non-zero values represent damaged regions
mask = cv2.imread('mask.jpg', 0)  # Assuming a binary mask

# Apply the inpainting algorithm (TELEA method in this example)
inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Display the result
cv2.imshow('Inpainted Image', inpainted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
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

:::{code} python
:caption: `line` example
start = (200, 100)
stop = (400, 100)
yellow = (0, 255, 255) # in BGR format

cv2.line(imageLine, start, stop, yellow, thickness=5, lineType=cv2.LINE_AA)
:::
::::

## medianBlur

(card-medianBlur)=
::::{card}
:header: medianBlur
:footer: [Documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga8d77eeba4167a3f62caf3dd0a71429c1)

Applies a median filter to an image.

The `cv2.medianBlur` function is used for reducing noise in an image, especially effective for removing "salt-and-pepper" noise. It works by replacing each pixel with the median of its surrounding pixel values, which reduces outliers without affecting the edges as much as other types of filters.

:::{code} python
:caption: `medianBlur` syntax
dst = cv2.medianBlur(src, ksize)
:::

|Parameters|Description|
|--|--|
|src| Input image (can be grayscale or color).|
|ksize| Aperture linear size. Must be an odd integer greater than 1, e.g., 3, 5, 7, etc.|

:::{code} python
:caption: `medianBlur` example
# Apply a median blur with a 5x5 kernel
dst = cv2.medianBlur(img, 5)

# Apply a median blur with a 7x7 kernel
dst = cv2.medianBlur(img, 7)
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

## multiply

(card-multiply)=
::::{card}
:header: multiply
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga979d898a58d7f61c53003e162e7ad89f)

Calculates the per-element product of two arrays or an array and a scalar.

The `cv2.multiply` function performs element-wise multiplication of two arrays or one array and a scalar. If two arrays are multiplied, they must have the same size and type. This operation is useful in applications like blending images, applying weights to pixels, or scaling images.

:::{code} python
:caption: `multiply` syntax
dst = cv2.multiply(src1, src2, scale=1.0, dtype=None)
:::

|Parameters|Description|
|--|--|
|src1| First input array (image)|
|src2| Second input array (image) or scalar|
|scale| Optional scale factor (default is 1.0)|
|dtype| Optional depth of the output array. If not specified, the depth is derived from the input arrays.|

:::{code} python
:caption: `multiply` example
# Multiply two images element-wise
result = cv2.multiply(image1, image2)

# Multiply an image by a scalar
result = cv2.multiply(image, 0.5)
:::
::::

## normalize

(card-normalize)=
::::{card}
:header: normalize
:footer: [Documentation](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga8d3f9fda4b189e067a4a3c897a107a77)

Normalizes the values of an array (image) to a specific range.

The `cv2.normalize` function is used to scale the pixel values of an image or the values in an array to a specified range. This operation is useful for enhancing the contrast or ensuring that values fall within a certain range.

:::{code} python
:caption: `normalize` syntax
dst = cv2.normalize(src, dst, alpha, beta, norm_type, dtype=None)
:::

|Parameters|Description|
|--|--|
|src| Input array or image|
|dst| Output array or image with normalized values|
|alpha| Lower range boundary for normalization|
|beta| Upper range boundary for normalization|
|norm_type| Type of normalization (`cv2.NORM_MINMAX`, `cv2.NORM_INF`, etc.)|
|dtype| Optional data type of the output array|

:::{code} python
:caption: `normalize` example
# Normalize the image pixel values to the range [0, 255]
normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# Display the original and normalized images
plt.subplot(121); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(122); plt.imshow(normalized_img, cmap='gray'); plt.title('Normalized')
plt.show()
:::
::::

## putText

(card-putText)=
::::{card}
:header: putText
:footer: [Documentation](https://docs.opencv.org/4.x/d6/d6e/group__

imgproc__draw.html#ga5126f47f883d730f633d74f07456c576)

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
img_specific = cv2.resize(img, dsize=(100, 200))
plt.imshow(img_specific)
plt.show()
:::
::::

## cv2.Sobel

(card-sobel)=
::::{card}
:header: cv2.Sobel
:footer: [Documentation](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html)

`cv2.Sobel` is used to compute the first, second, or higher-order image derivatives in the x or y direction. It is commonly used for edge detection by computing the gradient of the image intensity. This can help highlight regions of high spatial frequency, such as edges.

:::{code} python
:caption: `Sobel` syntax
dst = cv2.Sobel(src, ddepth, dx, dy, ksize, scale, delta, borderType)
:::

|Parameters|Description|
|--|--|
|src| Input image (can be grayscale or color)|
|ddepth| Desired depth of the output image (e.g., `cv2.CV_64F`, `cv2.CV_32F`, `cv2.CV_8U`)|
|dx| Order of the derivative in x direction (e.g., 1 for first derivative, 0 for no derivative in x)|
|dy| Order of the derivative in y direction (e.g., 1 for first derivative, 0 for no derivative in y)|
|ksize| Size of the extended Sobel kernel (must be 1, 3, 5, or 7)|
|scale| Optional scale factor for the computed derivative values|
|delta| Optional delta value added to the results|
|borderType| Pixel extrapolation method used at the image border (e.g., `cv2.BORDER_DEFAULT`)|

### Example:

:::{code} python
:caption: `Sobel` example
import cv2
import numpy as np

# Read the image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Sobel filter to detect horizontal edges
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# Apply Sobel filter to detect vertical edges
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Convert back to uint8 (8-bit image) for visualization
sobel_x = np.uint8(np.abs(sobel_x))
sobel_y = np.uint8(np.abs(sobel_y))

# Display the results
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
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

## threshold

(card-threshold)=
::::{card}
:header: threshold
:footer: [Documentation](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)

Applies a fixed-level threshold to each array element.

The function applies fixed-level thresholding to a multiple-channel array. The function is typically used to get a bi-level (binary) image out of a grayscale image or for removing noise. 

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

# Matplotlib

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

## hist

(card-hist)=
::::{card}
:header: hist
:footer: [Documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)

Creates a histogram to visualize the distribution of data.

The `plt.hist` function in Matplotlib generates a histogram, which is a graphical representation of the distribution of numerical data. It groups data points into bins and shows the frequency of data points in each bin.

:::{code} python
:caption: `hist` syntax
plt.hist(x, bins=None, range=None, density=False, weights=None, cumulative=False, **kwargs)
:::

|Parameters|Description|
|--|--|
|x| Input data to create the histogram.|
|bins| The number of bins or intervals for grouping the data points.|
|range| The lower and upper range of the bins.|
|density| If `True`, the histogram is normalized to form a probability density.|
|weights| An array of weights for the data points.|
|cumulative| If `True`, the histogram is cumulative.|
|**kwargs| Additional keyword arguments for customizing the histogram.|
|Return| The values of the histogram, the bin edges, and the patches used to draw the histogram.|

:::{code} python
:caption: `hist` example
data = np.random.randn(1000)
plt.hist(data, bins=30, color='blue', edgecolor='black')
plt.show()
:::
::::

## imshow

(card-imshow)=
::::

{card}
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

# NumPy

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

## linspace

(card-linspace)=
::::{card}
:header: linspace
:footer: [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

Generate evenly spaced numbers over a specified interval.

`np.linspace` returns evenly spaced numbers over a specified range. It is useful for creating numeric grids or intervals, such as time intervals or sample points for functions.

:::{code} python
:caption: `linspace` syntax
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
:::

|Parameters|Description|
|---|---|
|start|The starting value of the sequence.|
|stop|The end value of the sequence, unless `endpoint=False`.|
|num|Number of evenly spaced samples to generate. Defaults to 50.|
|endpoint|If `True` (default), `stop` is the last sample. If `False`, it is excluded.|
|retstep|If `True`, return (samples, step), where `step` is the spacing between samples.|
|dtype|The type of the output array. If `None`, infer the data type from inputs.|
|axis|The axis in the result along which the samples are stored. Default is 0.|

:::{code} python
:caption: `linspace` example
# Create 5 points between 0 and 1
np.linspace(0, 1, num=5)

# Create 5 points between 0 and 1, excluding the endpoint
np.linspace(0, 1, num=5, endpoint=False)
:::
::::

## fft

(card-fft)=
::::{card}
:header: fft
:footer: [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html)

Compute the one-dimensional discrete Fourier Transform.

`np.fft.fft` computes the one-dimensional **discrete Fourier Transform (DFT)** of a real or complex-valued input signal. It transforms the input data from the time or spatial domain into the frequency domain.

:::{code} python
:caption: `fft` syntax
np.fft.fft(a, n=None, axis=-1, norm=None)
:::

|Parameters|Description|
|---|---|
|a|Input array, can be complex.|
|n|Length of the transformed axis. If `n` is smaller than the length of `a`, the input is cropped. If `n` is larger, the input is padded with zeros. Defaults to `None`, meaning no padding or cropping.|
|axis|Axis over which to compute the FFT. Default is the last axis (-1).|
|norm|Normalization mode (`None` or `'ortho'`). Default is `None`.|

:::{code} python
:caption: `fft` example
# Compute the FFT of a simple sine wave
import numpy as np
t = np.linspace(0, 1, 500)
signal = np.sin(2 * np.pi * 5 * t)
fft_signal = np.fft.fft(signal)
:::
:::: 

## fftfreq

(card-fftfreq)=
::::{card}
:header: fftfreq
:footer: [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html)

Return the Discrete Fourier Transform sample frequencies.

`np.fft.fftfreq` computes the frequency bin centers in cycles per unit of the sample spacing for the Discrete Fourier Transform. It is often used in conjunction with `np.fft.fft` to interpret the frequency components of the FFT result.

:::{code} python
:caption: `fftfreq` syntax
np.fft.fftfreq(n, d=1.0)
:::

|Parameters|Description|
|---|---|
|n|Window length (the number of samples).|
|d|Sample spacing (inverse of the sampling rate). Defaults to 1.0.|

:::{code} python
:caption: `fftfreq` example
# Get the frequency bins for a signal of 500 samples with a sample spacing of 1/500
n = 500
sample_spacing = 1.0 / 500
frequencies = np.fft.fftfreq(n, d=sample_spacing)
:::
:::: 


## ifft

(card-ifft)=
::::{card}
:header: ifft
:footer: [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html)

Compute the one-dimensional inverse discrete Fourier Transform.

`np.fft.ifft` computes the **inverse discrete Fourier Transform (IDFT)**, transforming data from the frequency domain back to the time (or spatial) domain. It reconstructs the original signal from its frequency components.

:::{code} python
:caption: `ifft` syntax
np.fft.ifft(a, n=None, axis=-1, norm=None)
:::

|Parameters|Description|
|---|---|
|a|Input array, can be complex.|
|n|Length of the transformed axis. If `n` is smaller than the length of `a`, the input is cropped. If `n` is larger, the input is padded with zeros. Defaults to `None`.|
|axis|Axis over which to compute the IFFT. Default is the last axis (-1).|
|norm|Normalization mode (`None` or `'ortho'`). Default is `None`.|

:::{code} python
:caption: `ifft` example
# Compute the IFFT of a signal in the frequency domain
import numpy as np
fft_signal = np.fft.fft(np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500)))
recovered_signal = np.fft.ifft(fft_signal)
:::
:::: 

## meshgrid

(card-meshgrid)=
::::{card}
:header: meshgrid
:footer: [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html)

Create coordinate matrices from coordinate vectors.

`np.meshgrid` generates coordinate matrices (grids) from two or more coordinate vectors. It is useful for constructing 2D or 3D grids for function evaluations and visualizations.

:::{code} python
:caption: `meshgrid` syntax
np.meshgrid(*xi, indexing='xy', sparse=False, copy=True)
:::

|Parameters|Description|
|---|---|
|*xi|1-D arrays representing the coordinates of the grid points.|
|indexing|Indexing convention: 'xy' (default) for Cartesian, 'ij' for matrix indexing.|
|sparse|If `True`, return sparse matrices for memory efficiency.|
|copy|If `True`, return a new copy of the data.|

:::{code} python
:caption: `meshgrid` example
# Create a 2D grid
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
X, Y = np.meshgrid(x, y)

# Now X and Y represent a grid of coordinates
:::
::::

## fft2

(card-fft2)=
::::{card}
:header: fft2
:footer: [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html)

Compute the two-dimensional discrete Fourier Transform.

`np.fft.fft2` computes the 2D discrete Fourier Transform (DFT) of a real or complex-valued 2D input. It transforms the data from the spatial domain to the frequency domain.

:::{code} python
:caption: `fft2` syntax
np.fft.fft2(a, s=None, axes=(-2, -1), norm=None)
:::

|Parameters|Description|
|---|---|
|a|Input array, can be real or complex.|
|s|Shape of the output (optional). If not provided, the shape of `a` is used.|
|axes|Axes over which to compute the FFT. Default is the last two axes (-2, -1).|
|norm|Normalization mode. Can be `None` (default) or `'ortho'` for orthonormal normalization.|

:::{code} python
:caption: `fft2` example
# Compute the 2D FFT of a 5x5 matrix
x = np.random.rand(5, 5)
fft_values_2d = np.fft.fft2(x)

# Compute the 2D FFT of an image
import matplotlib.pyplot as plt
image = plt.imread('sample_image.jpg')
fft_image = np.fft.fft2(image)
:::
::::

## fftshift

(card-fftshift)=
::::{card}
:header: fftshift
:footer: [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html)

Shift the zero-frequency component to the center of the spectrum.

`np.fft.fftshift` shifts the zero-frequency component to the center of the array, which is useful for visualizing frequency spectra from the Fourier Transform. It reorders the output of `np.fft.fft` or `np.fft.fft2` so that the zero frequency is in the middle.

:::{code} python
:caption: `fftshift` syntax
np.fft.fftshift(x, axes=None)
:::

|Parameters|Description|
|---|---|
|x|Input array, can be real or complex.|
|axes|Axes over which to shift. If `None`, all axes are shifted.|

:::{code} python
:caption: `fftshift` example
# Compute the 2D FFT and shift the zero-frequency component to the center
fft_values_2d = np.fft.fft2(np.random.rand(5, 5))
fft_shifted = np.fft.fftshift(fft_values_2d)

# Plot the shifted magnitude spectrum
import matplotlib.pyplot as plt
plt.imshow(np.log(np.abs(fft_shifted) + 1), cmap='gray')
plt.title('2D FFT Shifted Magnitude Spectrum')
plt.show()
:::
::::

## ifft2

(card-ifft2)=
::::{card}
:header: ifft2
:footer: [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html)

Compute the two-dimensional inverse discrete Fourier Transform.

`np.fft.ifft2` computes the 2D inverse discrete Fourier Transform (IDFT). It converts a frequency-domain representation back into the spatial or time domain.

:::{code} python
:caption: `ifft2` syntax
np.fft.ifft2(a, s=None, axes=(-2, -1), norm=None)
:::

|Parameters|Description|
|---|---|
|a|Input array in the frequency domain (can be complex).|
|s|Shape of the output (optional). If not provided, the shape of `a` is used.|
|axes|Axes over which to compute the IFFT. Default is the last two axes (-2, -1).|
|norm|Normalization mode. Can be `None` (default) or `'ortho'` for orthonormal normalization.|

:::{code} python
:caption: `ifft2` example
# Compute the 2D inverse FFT of a frequency-domain representation
fft_values_2d = np.fft.fft2(np.random.rand(5, 5))
recovered_signal_2d = np.fft.ifft2(fft_values_2d)

# Plot the real part of the recovered signal
import matplotlib.pyplot as plt
plt.imshow(recovered_signal_2d.real, cmap='gray')
plt.title('Recovered 2D Signal from IFFT')
plt.show()
:::
::::

# SciPy

## dct

(card-dct)=
::::{card}
:header: dct
:footer: [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html)

Compute the Discrete Cosine Transform (DCT).

`scipy.fftpack.dct` computes the **1D Discrete Cosine Transform (DCT)** of a real-valued input signal. It transforms the data from the spatial domain to the frequency domain.

:::{code} python
:caption: `dct` syntax
scipy.fftpack.dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False)
:::

|Parameters|Description|
|---|---|
|x|Input array, must be real.|
|type|Type of the DCT (1, 2, 3, or 4). Default is 2.|
|n|Length of the transform (optional). If not provided, the length of `x` is used.|
|axis|Axis along which the DCT is computed. Default is the last axis.|
|norm|Normalization mode. Can be `None` (default) or `'ortho'` for orthonormal normalization.|
|overwrite_x|If `True`, the contents of `x` can be destroyed for efficiency. Default is `False`.|

:::{code} python
:caption: `dct` example
from scipy.fftpack import dct
import numpy as np

# Create a simple 1D signal
x = np.array([0.0, 1.0, 2.0, 3.0])

# Compute the DCT (Type-II, orthonormal)
dct_signal = dct(x, norm='ortho')
print(dct_signal)
:::
::::

## idct

(card-idct)=
::::{card}
:header: idct
:footer: [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.idct.html)

Compute the Inverse Discrete Cosine Transform (IDCT).

`scipy.fftpack.idct` computes the **1D Inverse Discrete Cosine Transform (IDCT)**, transforming data from the frequency domain back to the spatial domain.

:::{code} python
:caption: `idct` syntax
scipy.fftpack.idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False)
:::

|Parameters|Description|
|---|---|
|x|Input array, must be real.|
|type|Type of the IDCT (1, 2, 3, or 4). Default is 2.|
|n|Length of the inverse transform (optional). If not provided, the length of `x` is used.|
|axis|Axis along which the IDCT is computed. Default is the last axis.|
|norm|Normalization mode. Can be `None` (default) or `'ortho'` for orthonormal normalization.|
|overwrite_x|If `True`, the contents of `x` can be destroyed for efficiency. Default is `False`.|

:::{code} python
:caption: `idct` example
from scipy.fftpack import idct

# Compute the Inverse DCT to recover the original signal
recovered_signal = idct(dct_signal, norm='ortho')
print(recovered_signal)
:::
::::

## hadamard

(card-hadamard)=
::::{card}
:header: hadamard
:footer: [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hadamard.html)

Generate a Hadamard matrix.

`scipy.linalg.hadamard` constructs a **Hadamard matrix**, which is a square matrix whose elements are either +1 or -1, and the rows are orthogonal.

:::{code} python
:caption: `hadamard` syntax
scipy.linalg.hadamard(n, dtype=<class 'numpy.int32'>)
:::

|Parameters|Description|
|---|---|
|n|The size of the matrix. Must be a power of 2.|
|dtype|Data type of the result. Default is `numpy.int32`.|

:::{code} python
:caption: `hadamard` example
from scipy.linalg import hadamard

# Create an 8x8 Hadamard matrix
H = hadamard(8)
print(H)
:::
::::

# scikit-learn

## fit

(card-fit)=
::::{card}
:header: `fit` method (scikit-learn)
:footer: [Documentation](https://scikit-learn.org/stable/glossary.html#term-fit)

The `fit` method is used to train machine learning models in scikit-learn by adjusting model parameters based on input data. It takes in training data and updates the internal model parameters.

:::{code} python
:caption: `fit` method syntax
model.fit(X, y=None)
:::

|Parameters|Description|
|--|--|
|X| The input data used to fit the model, typically a 2D array-like structure of shape `(n_samples, n_features)`|
|y| The target labels (optional for unsupervised learning)|
|sample_weight| Optional weights for each sample, shape `(n_samples,)`|

### Example:

:::{code} python
:caption: `fit` example
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate random data
X = np.random.rand(100, 1)  # features
y = 3 * X.squeeze() + 2  # target with a linear relation

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the model parameters (intercept and slope)
print(f"Intercept: {model.intercept_}, Slope: {model.coef_}")
:::

### Usage:
- The `fit` method trains the model by learning patterns from the input data `X` and, in supervised cases, the corresponding target values `y`.
- This method is called before using model-specific methods like `predict` or `score`.

::::

## GaussianMixture

(card-gaussianmixture)=
::::{card}
:header: GaussianMixture
:footer: [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

`sklearn.mixture.GaussianMixture` is a clustering algorithm that models data as a mixture of several Gaussian distributions. It estimates the parameters of these distributions using the Expectation-Maximization (EM) algorithm.

:::{code} python
:caption: `GaussianMixture` syntax
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X)
:::

|Parameters|Description|
|--|--|
|n_components| The number of mixture components (i.e., clusters)|
|covariance_type| Type of covariance to use: 'full', 'tied', 'diag', or 'spherical'|
|max_iter| Maximum number of iterations to run the EM algorithm (default: 100)|
|tol| Convergence threshold (default: 1e-3)|
|random_state| Seed for random number generation (useful for reproducibility)|
|init_params| Method to initialize weights, means, and covariances ('kmeans' or 'random')|

### Methods:

|Method|Description|
|--|--|
|fit(X)| Fit the Gaussian mixture model to the data `X`|
|predict(X)| Predict the cluster labels for each data point in `X`|
|sample(n_samples)| Generate random samples from the fitted mixture model|
|score(X)| Compute the log-likelihood of the data under the model|

### Example:

:::{code} python
:caption: `GaussianMixture` example
import numpy as np
from sklearn.mixture import GaussianMixture

# Generate random data
X = np.random.rand(100, 2)

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Print the predicted labels
print(labels)
:::
::::

## predict

(card-predict)=
::::{card}
:header: `predict` method (scikit-learn)
:footer: [Documentation](https://scikit-learn.org/stable/glossary.html#term-predict)

The `predict` method is used to make predictions using a trained model in scikit-learn. It takes input data and returns the predicted output based on the model's learned parameters.

:::{code} python
:caption: `predict` method syntax
model.predict(X)
:::

|Parameters|Description|
|--|--|
|X| Input data of shape `(n_samples, n_features)`, where `n_samples` is the number of samples and `n_features` is the number of features for each sample.|

### Example:

:::{code} python
:caption: `predict` example
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate random data
X = np.random.rand(100, 1)  # features
y = 3 * X.squeeze() + 2  # target with a linear relation

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_test = np.array([[0.5], [1.0], [1.5]])
predictions = model.predict(X_test)

print("Predictions:", predictions)
:::

### Usage:
- The `predict` method uses the fitted model to predict outputs for new input data `X`.
- This method is typically called after training a model with the `fit` method.
- The returned predictions depend on the type of model (e.g., regression models return continuous values, classification models return class labels).
::::