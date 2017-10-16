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


# read in the images
# image1 will be our low pass image
image1 = np.float64(misc.imread('../images/dog.bmp', flatten=0, mode='RGB'))

# image2 will be our high pass image
image2 = np.float64(misc.imread('../images/cat.bmp', flatten=0, mode='RGB'))

def cross_correlate_2d(image, kernel):
    '''
    Like convolution but doesn't involve flipping the kernel.
    '''
    # normalize the image
    norm_image = (1.0/250)*(image)

    # output placeholder
    output = np.zeros(image.shape)

    # save image and kernel shapes, respectively
    (image_height, image_width) = image.shape[:2]
    (kernel_height, kernel_width) = kernel.shape[:2]

    # padding
    vertical_pad = (kernel_height-1)/2
    horizontal_pad = (kernel_width-1)/2

    # now let's pad the image
    padded_image = np.pad(image, pad_width=((vertical_pad, vertical_pad), (horizontal_pad, horizontal_pad)), mode='reflect')

    # loop through the image
    for i in range(0, image_height):
        for j in range(0, image_width):
            # the region of interest, the window for the kernel + image matrix
            window = padded_image[i:kernel_height+i, j:kernel_width+j]

            # for the total calculation
            total = []

            # loop through the smaller window
            for x in range(0, window.shape[0]):
                for y in range(0, window.shape[1]):
                    total.append(window[x, y] * kernel[x, y])

            # output location = the sum of the total of that window
            output[i, j] = sum(total)

    print("%s") % "We're here in CORRELATE"
    return output

def convolve2d(image, kernel, color):
    '''
    We use cross_correlate_2d() here and use the flipped kernel to convolve.
    '''
    # flip the kernel
    flip_kernel = np.flipud(np.fliplr(kernel))

    if color == 1:
        # blue
        image[:,:,0] = cross_correlate_2d(image[:,:,0], flip_kernel)

        # green
        image[:,:,1] = cross_correlate_2d(image[:,:,1], flip_kernel)

        # red
        image[:,:,2] = cross_correlate_2d(image[:,:,2], flip_kernel)

        return image

    else:
        return cross_correlate_2d(image, flip_kernel)

def gaussian_blur_kernel_2d(kernel_size, sigma):
    '''Produces a kernal of a given height and width which can then be passed to convolve_2d
    from above, along with an image to produce a blurred version of the image
    '''
    if kernel_size % 2 == 1:
        x, y = np.mgrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
        gmask = np.exp(-((x**2/float(kernel_size) + y**2/float(kernel_size))/(2 * sigma **2)))
        gmask = gmask / gmask.sum()
        return gmask

    else:
        raise ValueError("Kernal size cannot be even.")

# low pass: blurring
def low_pass(image):
    '''Removes the fine details from an image (blurs) ; will use convolve2d'''
    print("%s") % "We're here in LOW"
    kernel = gaussian_blur_kernel_2d(11, 5)
    low_pass_image = convolve2d(image, kernel, color=1)
    return low_pass_image

# high pass: sharpening
def high_pass(image):
    '''Removes the coarse details from an image (sharpens); will use convolve2d to blur the image'''
    print("%s") % "We're here in HIGH"
    # make a copy of the image to keep original + get a copy for blurring
    original = np.copy(image)
    copy = np.copy(image)
    blurred_image = low_pass(copy)
    alpha = 1
    high_pass_image = alpha * (image - blurred_image)
    return high_pass_image


# processing images
low_passed = low_pass(image1)
high_passed = high_pass(image2)

# hybridize the images
hybrid = low_passed + high_passed

# saving
misc.imsave("../images/output/testlow.png", low_passed)
misc.imsave("../images/output/testhigh.png", high_passed)
misc.imsave("../images/output/testhybrid.png", hybrid)

print("%s" % "Done")

# checks for grayscale images
# plt.subplot(1, 3, 1)
# plt.imshow(low_passed, cmap='gray')
# plt.title('Low pass')
# plt.subplot(1, 3, 2)
# plt.imshow(high_passed, cmap='gray')
# plt.title('High pass')
# plt.subplot(1, 3, 3)
# plt.imshow(hybrid, cmap='gray')
# plt.title('Hybrid')
# plt.show()
