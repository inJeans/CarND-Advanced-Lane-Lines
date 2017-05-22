from __future__ import print_function

import os
import glob

import cv2

import numpy as np
import matplotlib.pyplot as plt

from logger import LOGGER, set_up_logger
from image_correction import calibrate_camera, undistort_image_data, warp_perspective

# GLOBALS
# Chessboard dimensions
NX = 6
NY = 9 

# Pixel to meter conversion factors
Y_M_PER_PIX = 30/720 # meters per pixel in y dimension
X_M_PER_PIX = 3.7/700 # meters per pixel in x dimension

# Data directories
CALIBRATION_DIR = 'camera_cal'
TEST_DIR = 'test_images'
UNDISTORTED_DIR = 'undistorted_images'
if not os.path.isdir(UNDISTORTED_DIR):
    os.mkdir(UNDISTORTED_DIR)

def main():
    """ Main data processing pipeline
    """
    objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs = calibrate_camera(CALIBRATION_DIR,
                                                                          (NX, NY))

    undistort_image_data(TEST_DIR,
                         UNDISTORTED_DIR,
                         objpoints,
                         imgpoints,
                         mtx,
                         dist,
                         rvecs,
                         tvecs)

if __name__ == '__main__':
    set_up_logger()
    main()
