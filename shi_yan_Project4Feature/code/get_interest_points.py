import cv2
import numpy as np
import scipy.ndimage.filters as filters 
import scipy.misc as misc 
import matplotlib.pyplot as plt

def get_interest_points(image, feature_width, threshold): 
	response = hariss_corner(image)
	
	# let's get rid of the edges of the image 
	edge_pad = feature_width - 1 
	response[:edge_pad, :] = 0 
	response[:, :edge_pad] = 0 
	response[-edge_pad:, :] = 0 
	response[:, -edge_pad:] = 0 

	# threshold 
	thresholdvalue = response.max() * threshold
	responsethreshold = response > thresholdvalue
	responsethreshold = responsethreshold.astype('uint8')

	coords = np.tranpose(np.nonzero(responsethreshold))

	corners = np.ones(response.shape)
	interest_points_x = np.array([])
	interest_points_y = np.array([])

	for coord in coords:
		if corners[coord[0], coord[1]] == 1: 
			interest_points_x = np.append(interest_points_x, coord[0])
			interest_points_y = np.append(interest_points_y, coord[1])
			corners[(coord[0]-feature_width): (coord[0]+feature_width), 
					(coord[1]-feature_width): (coord[1]+feature_width)]

	return interest_points_x, interest_points_y # it is optional to return scale and orientation


def harris_corner(image): 
	""" only for grayscale images !!! """ 
	# deriving derivatives 
	derivative_x = filters.gaussian_filter(image, (3, 3), (0,1))
	derivative_y = filters.gaussian_filter(image, (3, 3), (1, 0))

	# computing the Harris matrix 
	xx = filters.gaussian_filter(derivative_x * derivative_x, 3)
	yy = filters.gaussian_filter(derivative_y * derivative_y, 3)
	xy = filters.gaussian_filter(derivative_x * derivative_y, 3)

	# determinant and trace 
	determinant = (xx * yy) - (xy ** 2)
	trace = (xx + yy) + .00000000000000001 # to swerve on the NaN values 

	return responses
	

