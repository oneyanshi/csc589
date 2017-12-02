import cv2
import numpy as np
import scipy.ndimage.filters as filters
from scipy import ndimage
import scipy.misc as misc 
import matplotlib.pyplot as plt 


def get_features(image, x, y, feature_width):

	features = np.array([])
	for row in range(x.shape[0]): 
		box = image[x[row]-feature_width/2 : x[row]+(feature_width+1)/2, 
					y[row]-feature_width/2 : y[row]+(feature_width+1)/2]

	columns = 0
	rows = 0
	phase_histogram_information = [] 
	while columns < feature_width: 
		while rows < feature_width: 
			sobel_x = np.array([])
			sobel_y = np.array([])
			smaller_box = box[rows:(rows+(feature_width/4)), columns:(columns+(feature_width/4))]
			
			# DO THE DERIVATIVE 
			sobel_x = cv2.Sobel(smaller_box, -1, 1, 0)
			sobel_y = cv2.Sobel(smaller_box, -1, 0, 1)
			
			phase_array = cv2.phase(sobel_x, sobel_y)
			phase_histogram_information.append(np.histogram(phase_array, 8))

			rows = rows + feature_width/4
		columns = columns + feature_width / 4

	# append these features together ; 128 dimensions 
	features = np.append(features, phase_histogram_information)
			
# % Your implementation does not need to exactly match the SIFT reference.
# % Here are the key properties your (baseline) descriptor should have:
# %  (1) a 4x4 grid of cells, each feature_width/4.
# %  (2) each cell should have a histogram of the local distribution of
# %    gradients in 8 orientations. Appending these histograms together will
# %    give you 4x4 x 8 = 128 dimensions.
# %  (3) Each feature should be normalized to unit length
# %
# % You do not need to perform the interpolation in which each gradient
# % measurement contributes to multiple orientation bins in multiple cells
# % As described in Szeliski, a single gradient measurement creates a
# % weighted contribution to the 4 nearest cells and the 2 nearest
# % orientation bins within each cell, for 8 total contributions. This type
# % of interpolation probably will help, though.

# % You do not have to explicitly compute the gradient orientation at each
# % pixel (although you are free to do so). You can instead filter with
# % oriented filters (e.g. a filter that responds to edges with a specific
# % orientation). All of your SIFT-like feature can be constructed entirely
# % from filtering fairly quickly in this way. 

# For filtering the image and computing the gradients, 
# you can either use the following functions or implement you own filtering code as you did in the second project:
#scipy.ndimage.sobel: Filters the input image with Sobel filter.
#scipy.ndimage.gaussian_filter: Filters the input image with a Gaussian filter.
#scipy.ndimage.filters.maximum_filter: Filters the input image with a maximum filter.
#scipy.ndimage.filters.convolve: Filters the input image with the selected filter.

# % You do not need to do the normalize -> threshold -> normalize again
# % operation as detailed in Szeliski and the SIFT paper. It can help, though.

# % Another simple trick which can help is to raise each element of the final
# % feature vector to some power that is less than one.











