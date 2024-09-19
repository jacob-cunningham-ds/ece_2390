import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
import pywt

# Function to load an image from a file path and interactively adjust the threshold
def finetune_threshold(image):
    """
    Takes an image (as a numpy array) and displays it with an interactive 
    thresholding widget. The user can adjust the threshold value with a slider, and 
    the binary thresholded image will be updated in real-time.

    The function provides both manual thresholding via the slider and automatically 
    displays the original image in grayscale alongside the thresholded version for comparison.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image as a numpy array. The image can be either grayscale or color (RGB/BGR).
        If the image is color, it will be automatically converted to grayscale for thresholding.

    How it works:
    -------------
    - If the image is in color, it is converted to grayscale for thresholding.
    - The thresholding slider allows you to set a threshold value between 0 and 255.
    - When the slider is adjusted, a binary image is created using the selected threshold value,
      and the thresholded image is displayed next to the original image in real time.

    Slider Interaction:
    -------------------
    - The threshold slider allows dynamic adjustment of the threshold value from 0 to 255.
    - When a new threshold value is selected, the binary thresholding result is immediately 
      updated and displayed.
    
    Visualization:
    --------------
    - The function shows two images side-by-side:
      - The original image in grayscale.
      - The thresholded (binary) image where pixels are either black or white based on 
        the threshold value.

    Example:
    --------
    To use the function, pass an image as a numpy array:

        finetune_threshold(image)

    This will display the interactive thresholding interface.
    """
    # Check if the image is in color, and convert to grayscale if necessary
    if len(image.shape) == 3:  # If the image has 3 channels (assumed to be RGB or BGR)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image

    # Convert the image to uint8 type if needed
    img_gray = np.array(img_gray, dtype=np.uint8)

    # Precompute Otsu's
    val_otsu, _ = cv2.threshold(img_gray, 127, 255, cv2.THRESH_OTSU)
    
    # Function to apply thresholding and display the result
    def apply_threshold(threshold_value):
        # Apply the threshold to the grayscale image
        _, binary_img = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Display the output
        fig, axs = plt.subplot_mosaic([['Gray', 'Binary'], ['Gray', 'Binary'], ['Histogram', 'Histogram']], figsize=(8, 8*2/3), layout='constrained')
        fig.suptitle('Threshold Finetuning')
        axs['Gray'].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
        axs['Gray'].set_title('Grayscale')
        axs['Binary'].imshow(binary_img, cmap='gray', vmin=0, vmax=255)
        axs['Binary'].set_title(f'Threshold: {threshold_value}')
        axs['Histogram'].hist(img_gray.ravel(), bins=75, alpha=0.7, color='gray', edgecolor='black')
        axs['Histogram'].axvline(x=threshold_value, color='r', linestyle='-', linewidth=2, label=f'Threshold: {threshold_value}')
        axs['Histogram'].axvline(x=val_otsu, color='g', linestyle='--', linewidth=2, label=f'Otsu: {val_otsu:.0f}')
        axs['Histogram'].legend()
        axs['Histogram'].set_title('Image Histogram')
        plt.show()

    # Create an interactive slider widget to control the threshold value
    threshold_slider = widgets.IntSlider(
        value=167,    # Initial value
        min=0,        # Minimum threshold value
        max=255,      # Maximum threshold value
        step=1,       # Step size for the slider
        description='Threshold:',
        continuous_update=False  # Update only when the slider is released
    )

    # Use interact to link the slider with the apply_threshold function
    _ = interact(apply_threshold, threshold_value=threshold_slider)

# Homework Problem 4
def median_filter_arbitrary(image, kernel_size):
    """
    Applies a median filter to an image using an arbitrary kernel size, ensuring that the 
    kernel size is odd to have a well-defined center pixel.

    The function works on both grayscale and color images. For color images, it applies the 
    median filter independently to each color channel (Red, Green, Blue). If an even-sized 
    kernel is provided, the function raises a ValueError.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image as a numpy array. The image can be either grayscale (2D array) or color (3D array).
        If the image is color, the median filter is applied separately to each channel (R, G, B).
    kernel_size : int or tuple of two ints
        The size of the kernel to be applied. If an int is provided, a square kernel of size
        (kernel_size, kernel_size) is used. If a tuple of two ints is provided, it represents the
        height and width of the kernel (kernel_height, kernel_width). Both values must be odd to have a 
        well-defined center pixel.

    How it works:
    -------------
    - For each pixel in the image, the function extracts a neighborhood defined by the kernel size.
    - It computes the median of the pixel values in the kernel and replaces the center pixel of the
      kernel with this median value.
    - The median filter is applied channel-by-channel for color images.

    Edge Handling:
    --------------
    - The function pads the image with edge values to handle pixels on the borders of the image,
      ensuring that every pixel, including border pixels, can be processed with the kernel.

    Constraints:
    ------------
    - The kernel size must be odd for both the height and width to have a well-defined center pixel.
    - If an even-sized kernel is provided, a ValueError is raised.

    Example:
    --------
    To apply a 5x5 median filter to a grayscale image:

        filtered_image = median_filter_arbitrary(image, kernel_size=5)

    To apply a 3x5 median filter to a color image:

        filtered_image = median_filter_arbitrary(image, kernel_size=(3, 5))

    Returns:
    --------
    filtered_image : numpy.ndarray
        The filtered image with the same dimensions as the input image. The image will be
        either grayscale or color, depending on the input.

    Raises:
    -------
    ValueError:
        If the kernel size provided is even (i.e., not odd for both height and width).
    """
    # Handle kernel size as either a single int or a tuple
    if isinstance(kernel_size, int):
        kernel_height, kernel_width = kernel_size, kernel_size
    elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        kernel_height, kernel_width = kernel_size
    else:
        raise ValueError("kernel_size must be an int or a tuple of two ints (height, width).")
    
    # Ensure the kernel size is odd
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("Kernel height and width must be odd to have a well-defined center pixel.")
    
    # Padding amounts (half the kernel size on each side)
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Process the image
    if len(image.shape) == 2:  # Grayscale image
        padded_image = np.pad(image, ((pad_height, pad_width)), mode='edge')
        filtered_image = np.zeros_like(image)

        for i in range(pad_height, padded_image.shape[0] - pad_height):
            for j in range(pad_width, padded_image.shape[1] - pad_width):
                kernel = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
                filtered_image[i - pad_height, j - pad_width] = np.median(kernel)

    elif len(image.shape) == 3:  # Color image
        rows, cols, channels = image.shape
        filtered_image = np.zeros((rows, cols, channels), dtype=np.uint8)

        for c in range(channels):
            channel = image[:, :, c]
            padded_channel = np.pad(channel, ((pad_height, pad_width)), mode='edge')
            
            # Correct the loop to stay within bounds
            for i in range(pad_height, padded_channel.shape[0] - pad_height):
                for j in range(pad_width, padded_channel.shape[1] - pad_width):
                    kernel = padded_channel[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]

                    # Make sure the indices are within bounds for the original image
                    if (i - pad_height) < filtered_image.shape[0] and (j - pad_width) < filtered_image.shape[1]:
                        filtered_image[i - pad_height, j - pad_width, c] = np.median(kernel)

    return filtered_image

# Homework Problem 5
def compress_image_haar(image, compression_rate=0.25):
    """
    Compresses a color image using the Haar wavelet by reducing the high/high-pass (HH) 
    component. The function applies the Haar wavelet transform on each color channel (R, G, B) 
    and reduces the HH component by a specified compression rate.

    The function works on color images and applies the Haar wavelet independently to each 
    color channel (Red, Green, Blue).

    Parameters:
    -----------
    image : numpy.ndarray
        The input image as a 3D array. The image must be a color image (BGR format with 3 channels).
    compression_rate : float, optional
        The percentage of the high/high-pass (HH) component to reduce. The default is 0.25 (25% reduction).

    How it works:
    -------------
    - The function first applies a 2D Haar wavelet transform to each of the three color channels (Red, Green, Blue).
    - It then reduces the high-frequency diagonal components (HH subband) by scaling them down based on the 
      provided compression rate.
    - After modifying the HH components, the function reconstructs the image by performing an inverse Haar wavelet 
      transform on each color channel.
    - The result is an image with some loss of high-frequency details, particularly in diagonal areas.

    Compression:
    ------------
    - The HH (high/high-pass) component represents diagonal high-frequency details in the image. Reducing this component 
      compresses the image by removing fine details, resulting in a smoother, compressed image.
    - The compression rate determines how much of the HH component is removed. A compression rate of 0.25 removes 25% of 
      the HH component.

    Example:
    --------
    To compress an image by removing 25% of the HH component:

        compressed_image = compress_image_haar(image, compression_rate=0.25)

    Returns:
    --------
    compressed_image : numpy.ndarray
        The compressed image with the same dimensions as the input image. The image will be color (3 channels).

    Raises:
    -------
    ValueError:
        If the compression rate is not between 0 and 1.
    """

    # Ensure compression rate is valid
    if not (0 < compression_rate < 1):
        raise ValueError("Compression rate must be a float between 0 and 1.")

    # Initialize a list to store the compressed channels
    compressed_channels = []

    # Perform wavelet transform on each color channel (R, G, B)
    for c in range(3):  # Loop through the color channels
        # Apply 2D Haar wavelet transform to the channel
        coeffs = pywt.dwt2(image[:, :, c], 'haar')
        LL, (LH, HL, HH) = coeffs  # LL: low-low, LH: low-high, HL: high-low, HH: high-high

        # Remove a portion of the HH component
        HH *= (1 - compression_rate)

        # Reconstruct the channel using the modified coefficients (with the reduced HH component)
        compressed_channel = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
        compressed_channels.append(compressed_channel)

    # Stack the channels back together and clip to valid pixel range [0, 255]
    compressed_image = np.stack(compressed_channels, axis=-1)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

    return compressed_image
