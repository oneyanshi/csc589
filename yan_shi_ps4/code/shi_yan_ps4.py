"""
author: yan shi
last update: oct. 29, 2017
"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

#code for generation of pyramids + interpolation + decimation
#provided by au professor bei xiao

# define our kernel, the 5-tap binomial filter
kernel = (1.0/256)*np.array([
                            [1, 4,  6,  4,  1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4,  6,  4,  1]])

def rgb_split(image):
    """
    to handle colors, we'll need to split the rgb channels
    colors come in as blue, green, red 
    """ 
    blue, green, red = cv2.split(image)
    return red, green, blue

def interpolate(image):
    """
    interpolates an image with upsampling rate, r = 2 
    """
    image_up = np.zeros((2*image.shape[0], 2*image.shape[1]))
    # upsample 
    image_up[::2, ::2] = image[:,:]
    # blur (we need to scale this up since kernel has unit area)
    # the length and width are doubled, so the area is quadrupled 

    return (4 * (ndimage.filters.convolve(image_up, kernel, mode='constant')))

def decimate(image): 
    """ 
    decimates an image with downsampling rate, r = 2
    """ 
    # downsample 
    image_blur = ndimage.filters.convolve(image, kernel, mode='constant')

    return image_blur[::2, ::2]

# construction of pryamids 
def pyramids(image): 
    """ 
    constructs Gaussian and Laplacian pyrmaids 
    parameters: 
        image: the original image (the base of the pyrmaid)
    returns: 
        GaussianPyramid: the gaussian pyrmaid 
        LaplacianPyramid: the laplacian pyrmaid 
    """ 
    GaussianPyramid = [image, ]
    LaplacianPyramid = []

    # build the gaussian pyrmaid 
    while image.shape[0] >= 2 and image.shape[1] >= 2: 
        image = decimate(image)
        GaussianPyramid.append(image)
    
    # build the laplacian pyramid
    for i in range(len(GaussianPyramid)-1): 
        # laplacian pyramids are found using the lvl of one gaussian level and 
        # the level of the next one 
        LaplacianPyramid.append(GaussianPyramid[i] - interpolate(GaussianPyramid[i + 1]))

    return GaussianPyramid[:-1], LaplacianPyramid

def construct_pyramid(image, pyramid):
    """ 
    for display purposes 
    parameters: 
        image: the original image 
        pyramid: the result from pyramids function 
    returns: 
        composite_image: the image containing the breakdown of the pyramid
    """ 
    rows, cols = image.shape
    composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
    composite_image[:rows, :cols] = pyramid[0]

    i_row = 0 
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols: cols + n_cols] = p 
        i_row += n_rows

    return composite_image

def reconstruct(laplacian_pyramid):
    # use rows and columns of the first laplacian pyramid 
    rows, cols = laplacian_pyramid[0].shape[0], laplacian_pyramid[0].shape[1]
    
    # create the output image 
    output_image = np.zeros((rows, cols), dtype=np.float64)
    
    # working backwards 
    for i in range(len(laplacian_pyramid)-1, 0, -1):
        # upsample laplacian_pyramid 
        laplacian = interpolate(laplacian_pyramid[i])

        # find the laplacian_pyramid associated w/ the next 
        laplacian_b = laplacian_pyramid[i-1]

        # add them together 
        beholder = laplacian + laplacian_b

        # we won't need these levels, so go ahead and get rid of them 
        laplacian_pyramid.pop()
        laplacian_pyramid.pop()

        # append them now 
        laplacian_pyramid.append(beholder)
        output_image = beholder
 
    return output_image 

def blending(laplacian_pyramid_1, laplacian_pyramid_2, gauss_pyramid_mask):
    """ 
    LS(i,j) = GR(I,j,)*LA(I,j) + (1-GR(I,j))*LB(I,j)
    """
    blend_pyramid = []
    for i in range(0, len(laplacian_pyramid_1)):
        # first half of the equation 
        first_half = (gauss_pyramid_mask[i] * laplacian_pyramid_1[i])

        # second half of the equation 
        second_half = ((1-gauss_pyramid_mask[i]) * (laplacian_pyramid_2[i]))

        # append the first + second to blend_pyrmaid 
        blend_pyramid.append(first_half + second_half)

    return blend_pyramid

def main():
    # """ Gray scale images """
    # image1_flat = misc.imread('../data/sun_flower.jpg', flatten=1)
    # image2_flat = misc.imread('../data/gao.jpg', flatten=1)
    # mask_flat = misc.imread('../data/gao_mask.jpg', flatten=1)

    # mask_flat = mask_flat.astype(float)/255

    # g_pyramid, l_pyramid = pyramids(image1_flat)
    # g_pyramid2, l_pyramid2 = pyramids(image2_flat)
    # g_pyramid_m, l_pyramid_m = pyramids(mask_flat)

    # g_pyramid_image = construct_pyramid(image1_flat, g_pyramid)
    # g_pyramid2_image = construct_pyramid(image2_flat, g_pyramid2) 

    # g_pyramid_m_image = construct_pyramid(mask_flat, g_pyramid_m)

    # l_pyramid_image = construct_pyramid(image1_flat, l_pyramid)
    # l_pyramid2_image = construct_pyramid(image2_flat, l_pyramid2) 

    # blended = blending(l_pyramid, l_pyramid2, g_pyramid_m)

    # reconstruct_image = reconstruct(blending(l_pyramid2, l_pyramid, g_pyramid_m))
    # plt.imshow(reconstruct_image, cmap='gray')
    # plt.show()

    # plt.imshow(g_pyramid_image, cmap='gray')
    # plt.show()
    # plt.imshow(g_pyramid2_image, cmap='gray')   
    # plt.show()
    # plt.imshow(l_pyramid_image, cmap='gray')
    # plt.show()
    # plt.imshow(l_pyramid2_image, cmap='gray')   
    # plt.show()

    """Color images""" 
    image1 = misc.imread('../data/sun_flower3.jpg', flatten=0, mode="RGB")
    image2 = misc.imread('../data/gao.jpg', flatten=0, mode="RGB")
    mask = misc.imread('../data/gao_mask.jpg', flatten=0, mode="RGB")

    # rgb split 
    red1, green1, blue1 = rgb_split(image1)
    red2, green2, blue2 = rgb_split(image2)
    redmask, greenmask, bluemask = rgb_split(mask)

    redmask = redmask.astype(float)/255
    greenmask = greenmask.astype(float)/255
    bluemask = bluemask.astype(float)/255

    # create the pyramids
    # image 1 
    GaussianPyramid_R, LaplacianPyramid_R = pyramids(red1)
    GaussianPyramid_G, LaplacianPyramid_G = pyramids(green1)
    GaussianPyramid_B, LaplacianPyramid_B = pyramids(blue1)

    # image 2
    GaussianPyramid_2_R, LaplacianPyramid_2_R = pyramids(red2)
    GaussianPyramid_2_G, LaplacianPyramid_2_G = pyramids(green2)
    GaussianPyramid_2_B, LaplacianPyramid_2_B = pyramids(blue2)

    # mask 
    gauss_mask_R, laplacian_mask_R = pyramids(redmask)
    gauss_mask_G, laplacian_mask_G = pyramids(greenmask)
    gauss_mask_B, laplacian_mask_B = pyramids(bluemask)

    # output image 
    output_R = reconstruct(blending(LaplacianPyramid_2_R, LaplacianPyramid_R, gauss_mask_R))
    output_G = reconstruct(blending(LaplacianPyramid_2_G, LaplacianPyramid_G, gauss_mask_G)) 
    output_B = reconstruct(blending(LaplacianPyramid_2_B, LaplacianPyramid_B, gauss_mask_B))

    # put it all together now
    result = np.zeros(image1.shape, dtype=np.float64)
    temporary = [] 
    temporary.append(output_B)
    temporary.append(output_G)
    temporary.append(output_R)
    result = cv2.merge(temporary, result)

    # save the image 
    #misc.imsave("../data/output/blend_gaoflower_part2_image.png", result)

    # rows, cols = image1.shape[:2]
    # split down the middle type of way
    # bad_result = np.hstack((image1[:,:cols/2], image2[:,cols/2:]))
    # misc.imsave("../data/output/amateur_blend.png", bad_result)

if __name__ == '__main__': 
    main()