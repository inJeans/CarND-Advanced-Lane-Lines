import cv2

import numpy as np

# from image_processing import

def find_window_centroids(warped, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/10):,:int(warped.shape[1]/2)], axis=0)
    np.convolve(window,l_sum)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    l_coords = (l_center, warped.shape[0]/10)
    r_sum = np.sum(warped[int(3*warped.shape[0]/10):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    r_coords = (r_center, warped.shape[0]/10)
    
    # Add what we found for the first layer
    window_centroids.append((l_coords,r_coords))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        l_coords = (l_center, (level+0.5)*window_height)
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        r_coords = (r_center, (level+0.5)*window_height)
        # Add what we found for that layer
        window_centroids.append((l_coords,r_coords))

    return window_centroids

def fit_lines(binary_warped,
	          left_line,
	          right_line):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.floor(binary_warped.shape[0]/2).astype(np.int):,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]).astype(np.int)
        good_right_inds = (((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]).astype(np.int)
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    try:
    	left_fit = np.polyfit(lefty, leftx, 2)
    except TypeError:
    	left_fit = [np.array(None)]
    try:
    	right_fit = np.polyfit(righty, rightx, 2)   
    except TypeError:
    	right_fit = [np.array(None)]


    left_line.update(left_fit,
                     right_fit,
                     leftx,
                     lefty,
                     binary_warped.shape)
    right_line.update(right_fit,
                      left_fit,
                      rightx,
                      righty,
                      binary_warped.shape)

    return

def visualise_lanes(plot_image,
                    binary_warped, 
                    left_line,
                    right_line,
                    Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    left_fit = left_line.best_fit
    right_fit = right_line.best_fit

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (plot_image.shape[1], plot_image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(plot_image, 1, newwarp, 0.3, 0)

    position = 0.5 * (left_fitx[-1] + right_fitx[-1] - binary_warped.shape[1])
    position = position * left_line.x_m_per_pix
    
    if position > 0:
    	relative_position = "right"
    else:
    	relative_position = "left"
    position_string = "Position: {0: 8.3f} m {1} of center".format(position, relative_position)
    curvature = 0.5 * (left_line.radius_of_curvature + right_line.radius_of_curvature)

    font = cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.putText(result,
    	                 position_string,
    	                 (50, plot_image.shape[0]-50), 
    	                 font, 
    	                 1,
    	                 (255,255,255),
    	                 2,
    	                 cv2.LINE_AA)
    result = cv2.putText(result,
    	                 "Curvature = {:8.3f} m".format(curvature),
    	                 (50, 50), 
    	                 font, 
    	                 1,
    	                 (255,255,255),
    	                 2,
    	                 cv2.LINE_AA)
    
    return result
