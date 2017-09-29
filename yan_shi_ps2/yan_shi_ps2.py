'''
Author: Yan Shi
Last Updated: 9/14/2017
Problem Set 1 | Prof Bei Xiao
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import misc
from scipy import ndimage

# ===  Problem 1: Warm Up ===
# # loading the images
# image1 = np.float64(misc.imread('images/peppers.png', flatten=1, mode='F'))
# image2 = np.float64(misc.imread('images/cheetah.png', flatten=1, mode='F'))
#
# # blur the images
# gau_image1 = ndimage.gaussian_filter(image1, 7)
# gau_image2 = ndimage.gaussian_filter(image2, 7)
#
# # computing dft of the image
# dft_image1 = np.fft.fft2(image1)
# dft_image2 = np.fft.fft2(image2)
#
# # computing dft of the blurred image
# dft_image1_blur = np.fft.fft2(gau_image1)
# dft_image2_blur = np.fft.fft2(gau_image2)
#
# # magnitudes
# # need to shift the images first
# # shifts of originals
# dft_image1_shift = np.fft.fftshift(dft_image1)
# dft_image2_shift = np.fft.fftshift(dft_image2)
#
# # shifts of blurred images
# dft_image1_blur_shift = np.fft.fftshift(dft_image1_blur)
# dft_image2_blur_shift = np.fft.fftshift(dft_image2_blur)
#
# # original images magnitudes
# magnitude_spectrum_image1 = np.log(np.abs(dft_image1_shift))
# magnitude_spectrum_image2 = np.log(np.abs(dft_image2_shift))
#
# # blur image magnitudes
# magnitude_spectrum_image1_blur = np.log(np.abs(dft_image1_blur_shift))
# magnitude_spectrum_image2_blur = np.log(np.abs(dft_image2_blur_shift))
#
# # for image1
# plt.figure(1, figsize=(15, 5))
# plt.subplot(2, 4, 1)
# plt.title('original', fontsize=10)
# plt.imshow(image1, cmap=plt.cm.gray)
# plt.axis('off')
# plt.subplot(2, 4, 2)
# plt.title('original magnitude spectrum', fontsize=10)
# plt.imshow(magnitude_spectrum_image1, cmap=plt.cm.gray)
# plt.axis('off')
# plt.subplot(2, 4, 3)
# plt.title('blurred', fontsize=10)
# plt.imshow(gau_image1, cmap=plt.cm.gray)
# plt.axis('off')
# plt.subplot(2, 4, 4)
# plt.title('blurred magnitude spectrum', fontsize=10)
# plt.imshow(magnitude_spectrum_image1_blur, cmap=plt.cm.gray)
# plt.axis('off')
#
# # for image2
# plt.subplot(2, 4, 5)
# plt.title('original2', fontsize=10)
# plt.imshow(image2, cmap=plt.cm.gray)
# plt.axis('off')
# plt.subplot(2, 4, 6)
# plt.title('original2 magnitude spectrum', fontsize=10)
# plt.imshow(magnitude_spectrum_image2, cmap=plt.cm.gray)
# plt.axis('off')
# plt.subplot(2, 4, 7)
# plt.title('blurred', fontsize=10)
# plt.imshow(gau_image2, cmap=plt.cm.gray)
# plt.axis('off')
# plt.subplot(2, 4, 8)
# plt.title('blurred2 magnitude spectrum', fontsize=10)
# plt.imshow(magnitude_spectrum_image2_blur, cmap=plt.cm.gray)
# plt.axis('off')
#
# plt.subplots_adjust(wspace=-.5, hspace=.5)
# plt.show()

# # for image1
# plt.figure(1, figsize=(20, 4))
# plt.subplot(1, 4, 1)
# plt.title('original', fontsize=15)
# plt.imshow(image1, cmap=plt.cm.gray)
# plt.subplot(1, 4, 2)
# plt.title('original magnitude spectrum', fontsize=15)
# plt.imshow(magnitude_spectrum_image1, cmap=plt.cm.gray)
# plt.subplot(1, 4, 3)
# plt.title('blurred', fontsize=15)
# plt.imshow(gau_image1, cmap=plt.cm.gray)
# plt.subplot(1, 4, 4)
# plt.title('blurred magnitude spectrum', fontsize=15)
# plt.imshow(magnitude_spectrum_image1_blur, cmap=plt.cm.gray)
# plt.show()
#
# # for image2
# plt.figure(2, figsize=(20, 4))
# plt.subplot(1, 4, 1)
# plt.title('original', fontsize=15)
# plt.imshow(image2, cmap=plt.cm.gray)
# plt.subplot(1, 4, 2)
# plt.title('original magnitude spectrum', fontsize=15)
# plt.imshow(magnitude_spectrum_image2, cmap=plt.cm.gray)
# plt.subplot(1, 4, 3)
# plt.title('blurred', fontsize=15)
# plt.imshow(gau_image2, cmap=plt.cm.gray)
# plt.subplot(1, 4, 4)
# plt.title('blurred magnitude spectrum', fontsize=15)
# plt.imshow(magnitude_spectrum_image2_blur, cmap=plt.cm.gray)
# plt.show()

# ===Problem 2: Histogram equilization ===
# low_contrast_image = np.float64(misc.imread('images/lowcontrast.jpg', flatten=1))
# image_hist, bins = np.histogram(low_contrast_image.flatten(), 256, [0,256], density=True)
# cdf = image_hist.cumsum()
# cdf_normalized = 255 * cdf / cdf[-1]
# new_image = np.interp(low_contrast_image.flatten(), bins[:-1], cdf_normalized)
# new_image = new_image.reshape(low_contrast_image.shape)
#
# plt.figure(figsize=(10, 5))
# plt.subplot(2, 2, 1)
# plt.hist(low_contrast_image.flatten(), 256, [0, 256])
# plt.xlim([0, 256])
# plt.title('original histogram')
# plt.subplots_adjust(hspace=.5)
# plt.subplot(2, 2, 2)
# plt.imshow(low_contrast_image, cmap=plt.cm.gray)
# plt.axis('off')
# plt.title('original')
#
# plt.subplot(2, 2, 3)
# plt.title('equalized image histogram')
# plt.hist(new_image.flatten(), 256, [0, 256])
# plt.xlim([0, 256])
# plt.subplot(2, 2, 4)
# plt.imshow(new_image, cmap=plt.cm.gray)
# plt.title('equalized')
# plt.axis('off')
# plt.show()

''' Problem 3:  Separable filters
Implement convolution with a separable kernel. The input should be grayscale or color image along
with the horizontal and vertical kernels.

Please include examples of Gaussian filter, box filter, and Sobel filter.'''
# gaussian filter

# box filter

# sobel filter

''' problem 3 part b
Make sure you support the padding mechanisms described in class and your textbook (chapter 3.2).
You will need this functionality for some of the later exercise. Please specify the kernels you used
and display the original image and the output images after each horizontal and vertical operation.

Here are a few padding options:
    To compensate for this, a number of alternative padding or extension modes have been developed:
    zero: set all pixels outside the source image to 0 (a good choice for alpha-matted cutout images);
    constant (border color): set all pixels outside the source image to a specified border value;
    clamp: (replicate or clamp to edge): repeat edge pixels indefinitely;
    (cyclic) wrap (repeat or tile); loop "around" the image in a torodial configuration;'''


# === Problem 4 ===
# zebra_image = np.float64(misc.imread('images/zebra.png', flatten=1, mode='F'))
#
# def find_edges(img):
#     '''find_edges(img) is a function that finds the total edges of an image.
#     Returns the x-direction of the sobel filter, the y-direction of the sobel filter,
#     and the total edges added together based on the x and y direction sobel filter.'''
#
#     image_x = np.zeros(img.shape)
#     image_x = ndimage.sobel(img, axis=1, mode='constant')
#
#     image_y = np.zeros(img.shape)
#     image_y = ndimage.sobel(img, axis=0, mode='constant')
#
#     total_edges = image_x + image_y
#
#     return image_x, image_y, total_edges
#
# image_x, image_y, edges = find_edges(zebra_image)
#
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(image_x, cmap=plt.cm.gray)
# plt.title('x-axis edges')
# plt.axis('off')
# plt.subplot(1, 3, 2)
# plt.imshow(image_y, cmap=plt.cm.gray)
# plt.title('y-axis edges')
# plt.axis('off')
# plt.subplot(1, 3, 3)
# plt.imshow(edges, cmap=plt.cm.gray)
# plt.title('all edges')
# plt.axis('off')
# plt.show()
