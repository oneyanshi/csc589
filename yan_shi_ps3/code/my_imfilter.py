# -*- coding: utf-8 -*-
'''
Author: Yan Shi
Last updated: October 12, 2017
Problem Set 3 | Prof. Bex Xiao | Intro to Computer Vision
'''

import numpy as np
import math
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc
from scipy import signal
import scipy.stats as st
from scipy.ndimage import filters
import cv2

'''
    image1 will be our low pass image
    image2 will be our high pass image
'''

# read in the images
image1 = np.float64(misc.imread('../images/marilyn.bmp', flatten=1))
image2 = np.float64(misc.imread('../images/einstein.bmp', flatten=1))

def convolve2d(image, kernel):
    '''
    Inputs: image, kernel
    Outputs: an image that has gone through convolution
    '''
    # normalize the image
    norm_image = (1.0/250)*(image)

    # output placeholder
    output = np.zeros(image.shape)

    # save image and kernel shapes, respectively
    (image_height, image_width) = image.shape[:2]

    # flip the kernel
    kernel = np.flipud(np.fliplr(kernel))

    # some nice padding
    pad = (norm_image.shape[1] - 1) / 2

    # we use cv2 here to make a border
    norm_image =cv2.copyMakeBorder(norm_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    for y in np.arange(pad, image_height + pad):
        for x in np.arange(pad, image_width + pad):
            # we try to skip over the padded areas
            region_of_interest = norm_image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # element-wise multiplcation the region of interest and the kernel
            total = (region_of_interest * kernel).sum()

            # place the total of that region into the
            output[y - pad, x - pad] = total

    return output

def gaussian_blur_kernel_2d(kernel_size, sigma):
    '''Produces a kernal of a given height and width which can then be passed to convolve_2d
    from above, along with an image to produce a blurred version of the image
    '''
    if kernel_size % 2 == 1:
        nrows = kernel_size
        kernel_array = np.zeros((nrows, nrows))
        center_x, center_y = ((nrows + 1.) / 2., (nrows + 1.) / 2.)

        # mesh grid creation
        x = np.linspace(0, nrows, nrows)
        X, Y = np.meshgrid(x, x)

        gmask = np.exp(-1.0 * ((X - center_x)**2. + (Y-center_y)**2.) / (2. * sigma**2.))
        return gmask

    else:
        raise ValueError("Kernal size cannot be even.")

#  == low pass: blurring ==
def low_pass(image):
    '''Removes the fine details from an image (blurs) ; will use convolve2d'''
    low_pass_image = convolve2d(image, gaussian_blur_kernel_2d(image.shape[1], 5))
    #misc.imsave("../images/output/low_pass_marilyn.png", np.real(low_pass_image))
    return low_pass_image

#  == high pass: sharpening ==
def high_pass(image):
    '''Removes the coarse details from an image (sharpens); will use convolve2d to blur the image'''
    blurred_image = convolve2d(image, gaussian_blur_kernel_2d(image.shape[1], 1))
    blurred_image2 = convolve2d(blurred_image, (gaussian_blur_kernel_2d(image.shape[1], 1)))
    alpha = 5

    high_pass_image = blurred_image + alpha * (blurred_image - blurred_image2)
    #misc.imsave("../images/output/high_pass_einstein.png", np.real(high_pass_image))
    return high_pass_image

low_passed = low_pass(image1)
high_passed = high_pass(image2)
hybrid = low_passed + high_passed
misc.imsave("../images/output/hybrid_image_marilyn_einstein.png", np.real(hybrid))
plt.subplot(1, 3, 1)
plt.imshow(low_passed, cmap='gray')
plt.title('Low pass')
plt.subplot(1, 3, 2)
plt.imshow(high_passed, cmap='gray')
plt.title('High pass')
plt.subplot(1, 3, 3)
plt.imshow(hybrid, cmap='gray')
plt.title('Hybrid')
plt.show()
