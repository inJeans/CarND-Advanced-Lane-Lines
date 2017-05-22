import glob
import os

import cv2

import numpy as np

from logger import LOGGER

def undistort_image_data(test_dir,
                         undistorted_dir,
                         objpoints,
                         imgpoints,
                         mtx,
                         dist,
                         rvecs,
                         tvecs):
    total_error = 0.

    test_image_path = os.path.join(test_dir,
                                   '*.jpg')
    images = glob.glob(test_image_path)

    for file_name in images:
        image = cv2.imread(file_name)
        error = undistort_image(file_name,
                                undistorted_dir,
                                image,
                                mtx,
                                dist,
                                rvecs,
                                tvecs,
                                objpoints,
                                imgpoints,)
        total_error += error
    
    LOGGER.info("The average image error is {}".format(total_error / len(images)))

def undistort_image(file_name,
                    save_directory,
                    image,
                    matrix,
                    dist,
                    rvecs,
                    tvecs,
                    objpoints=None,
                    imgpoints=None):
    height, width = image.shape[:2]
    
    # Refine camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix,
                                                           dist,
                                                           (width, height),
                                                           1,
                                                           (width,height))
    
    # Undistort
    undistorted = cv2.undistort(image,
                                matrix,
                                dist,
                                None,
                                new_camera_matrix)

    # crop the image
    x, y, new_width, new_height = roi
    undistorted = undistorted[y:y+new_height, x:x+new_width]
    
    # Save image
    undistorted_file_name = os.path.join(save_directory,
                                         file_name.split('\\')[-1])
    LOGGER.info(undistorted_file_name)
    cv2.imwrite(undistorted_file_name,
                undistorted)

    error = None
    if objpoints is not None and imgpoints is not None:
        error = check_error(matrix,
                            dist,
                            rvecs,
                            tvecs,
                            objpoints,
                            imgpoints)
    
    return error

def check_error(matrix,
                dist,
                rvecs,
                tvecs,
                objpoints,
                imgpoints):
    total_error = 0.
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i],
                                          rvecs[i],
                                          tvecs[i], 
                                          matrix,
                                          dist)
        error = cv2.norm(imgpoints[i],
                         imgpoints2,
                         cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)
    
    return mean_error

def calibrate_camera(calibration_dir,
                     chessboard_dim):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    calibration_image_path = os.path.join(calibration_dir,
                                          '*.jpg')
    images = glob.glob(calibration_image_path)

    for file_name in images:
        LOGGER.info("Processing {}...".format(file_name))
        objp, imgp = find_corners(file_name,
                                  chessboard_dim[0],
                                  chessboard_dim[1])

        objpoints.append(objp)
        imgpoints.append(imgp)

    example_image = cv2.imread(images[0])
    example_shape = example_image.shape[::-2]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       example_shape,
                                                       None,
                                                       None)

    return objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs
    # return None, None, None, None, None, None, None

def find_corners(file_name,
                 nx,
                 ny):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    refined_corners = None

    img = cv2.imread(file_name)
    grey = cv2.cvtColor(img,
                        cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(grey,
                                             (nx, ny),
                                             None)
    # If found, add object points, image points (after refining them)
    if ret is True:
        LOGGER.info("... corners found")
        # Refine corner locations
        refined_corners = cv2.cornerSubPix(grey,
                                           corners,
                                           (11, 11),
                                           (-1, -1),
                                           criteria)

    # If you don't find any corners, try again with fewer rows
    if ret is False and nx == 6:
        LOGGER.debug("Trying again with fewer rows")
        objp, refined_corners = find_corners(file_name,
                                             nx-1,
                                             ny)
    # If you don't find any corners, try again with fewer columns
    if ret is False and nx == 5:
        LOGGER.debug("Trying again with fewer columns")
        objp, refined_corners = find_corners(file_name,
                                             nx,
                                             ny-1)

    return objp, refined_corners

def warp_perspective(image):
    # Hard coded destination points. Manually found by inspecting image
    # Test images
    # src = np.float32([[496,434], [716,434], [300,567], [924,567]])
    # dst = np.float32([[100,600], [image.shape[0]-100,600], [100,image.shape[1]], [image.shape[0]-100,image.shape[1]]])
    # Project video
    # src = np.float32([[493,541], [857,541], [347,665], [1102,665]])
    # dst = np.float32([[100,1100], [image.shape[0]-100,1100], [100,image.shape[1]], [image.shape[0]-100,image.shape[1]]])
#     # Challenge video
    # src = np.float32([[466,584], [909,584], [351,667], [1046,667]])
    # dst = np.float32([[100,1100], [image.shape[0]-100,1100], [100,image.shape[1]], [image.shape[0]-100,image.shape[1]]])
#     # Harder challenge video
    src = np.float32([[480,515], [763,515], [268,664], [956,664]])
    dst = np.float32([[100,600], [image.shape[0]-100,600], [100,image.shape[1]], [image.shape[0]-100,image.shape[1]]])
    
    M = cv2.getPerspectiveTransform(src,
                                    dst)
    Minv = cv2.getPerspectiveTransform(dst,
                                       src)
    warped = cv2.warpPerspective(image,
                                 M,
                                 image.shape[:2],
                                 flags=cv2.INTER_LINEAR)
    
    return warped, Minv