import cv2

import numpy as np

def apply_colour_mask(image,
                      threshold_low,
                      threshold_high):
    binary_image = np.zeros_like(image[:,:,0])
    
    binary_image[(image[:, :, 0] >= threshold_low[0]) & (image[:, :, 0] <= threshold_high[0]) & \
                 (image[:, :, 1] >= threshold_low[1]) & (image[:, :, 1] <= threshold_high[1]) & \
                 (image[:, :, 2] >= threshold_low[2]) & (image[:, :, 2] <= threshold_high[2])] = 1
    
    return binary_image

def abs_sobel_thresh(img,
                     orient='x',
                     sobel_kernel=3,
                     thresh=(0, 255)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return binary_output

def mag_threshold(image,
                  sobel_kernel=3,
                  mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(image,
                  sobel_kernel=3,
                  thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def detect_lines(image):
    """ Following approach taken in this tutorial 
        https://medium.com/towards-data-science/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_hsv_low  = np.array([ 80, 80, 200])
    yellow_hsv_high = np.array([ 120, 255, 255])
    yellow_binary = apply_colour_mask(hsv_image,
                                      yellow_hsv_low,
                                      yellow_hsv_high)
    
    white_hsv_low  = np.array([  0,   0,   150])
    white_hsv_high = np.array([ 150,  80, 255])
    white_binary = apply_colour_mask(hsv_image,
                                     white_hsv_low,
                                     white_hsv_high)
    
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)
    sat_hls_low  = np.array([   0,    0, 80])
    sat_hls_high = np.array([ 255,  255, 255])
    sat_binary = apply_colour_mask(hls_image,
                                   sat_hls_low,
                                   sat_hls_high)
    
    colour_binary = np.zeros_like(white_binary)
    colour_binary[((yellow_binary == 1) | (white_binary == 1)) & (sat_binary == 1)] = 1
    
    # Convert to HLS color space and separate the V channel
    l_channel = hls_image[:,:,1]
    s_channel = hls_image[:,:,2]
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=9, thresh=(10, 150))
    grady = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=9, thresh=(20, 75))
    mag_binary = mag_threshold(s_channel, sobel_kernel=9, mag_thresh=(15, 100))
    dir_binary = dir_threshold(s_channel, sobel_kernel=9, thresh=(0.75, 1.5))
    
    sobel_binary = np.zeros_like(colour_binary)
    sobel_binary[((gradx == 1) | (grady == 1)) & ((mag_binary == 1))] = 1

    combined = np.zeros_like(sobel_binary)
    combined[(sobel_binary == 1) | (colour_binary == 1)] = 1
    
    return combined