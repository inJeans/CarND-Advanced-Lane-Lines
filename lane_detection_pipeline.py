from __future__ import print_function

import os
import glob

import cv2

from moviepy.editor import VideoClip

import numpy as np
import matplotlib.pyplot as plt

from logger import LOGGER, set_up_logger
from image_correction import calibrate_camera, undistort_image_data, warp_perspective
from image_preprocessing import detect_lines
from detect_lanes import fit_lines, visualise_lanes
from line import Line

# GLOBALS
# Chessboard dimensions
NX = 6
NY = 9 

# Pixel to meter conversion factors
Y_M_PER_PIX = 15./1240 # meters per pixel in y dimension
X_M_PER_PIX = 3.7/605 # meters per pixel in x dimension

# Data directories
CALIBRATION_DIR = 'camera_cal'
TEST_DIR = 'test_images'
UNDISTORTED_DIR = 'undistorted_images'
if not os.path.isdir(UNDISTORTED_DIR):
    os.mkdir(UNDISTORTED_DIR)

# CAP = cv2.VideoCapture('project_video.mp4')
# CAP = cv2.VideoCapture('challenge_video.mp4')
CAP = cv2.VideoCapture('harder_challenge_video.mp4')

ALPHA = 0.3
LEFT_LINE = Line(X_M_PER_PIX,
                 Y_M_PER_PIX,
                 ALPHA)
RIGHT_LINE = Line(X_M_PER_PIX,
                  Y_M_PER_PIX,
                  ALPHA)

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

    # animation = VideoClip(make_frame, duration=50)
    # animation = VideoClip(make_frame, duration=16)
    animation = VideoClip(make_frame, duration=47)
    # export as a video file
    # animation.write_videofile("project_animation.mp4", fps=25)
    # animation.write_videofile("challenge_animation.mp4", fps=25)
    animation.write_videofile("harder_challenge_animation.mp4", fps=25)

    CAP.release()

def make_frame(t):
    # global alpha
    if CAP.isOpened():
        ret, frame = CAP.read()

        plot_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        warped_image, Minv = warp_perspective(plot_image)
        binary_warped = detect_lines(warped_image)

        fit_lines(binary_warped,
                  LEFT_LINE,
                  RIGHT_LINE)
        
        detected_lanes = visualise_lanes(plot_image,
                                         binary_warped, 
                                         LEFT_LINE,
                                         RIGHT_LINE,
                                         Minv)
        
        return detected_lanes

if __name__ == '__main__':
    set_up_logger()
    main()
