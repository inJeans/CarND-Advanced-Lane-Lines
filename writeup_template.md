## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/thresholded.png "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example.png "Output"
[video1]: ./project_animation.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Using OpenCV's `findChessboardCorners` I was able to build a list of `objpoints` and `imgpoints` by iterating through the directory of given chessboard images. I further refined the location of the chessborad corners using `cornerSubPix`. All the images in the test directory were able to be used by programatically reducing the number of corners I would search for when `findChessboardCorners` returned `False`. Once I had built these lists of points I pushed them through the `calibrateCamera` function to get the camera calibration and distortion coefficients.

All of the heavy lifting for the camera calibration is (funilly enough) handled by my `calibrate_camera` function located in the `image_correction.py` file.

Images could then be undistorted by first refining the camera matrix with `getOptimalNewCameraMatrix` and then using the `undistort` function. A resulting example is shown below.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (the code for which is in `image_preprocessing`).  Here's an example of my output for this step.

![alt text][image3]

For the colour thresholds I apply both a yellow and a white filter in the HSV colour space, I also apply a saturation threshold in the HLS colour space. I `OR` the two colour thresholds together and then `AND` the result of that with the saturation threshold. So I only require a line to be yellow or white. The gradients are thresholded i the `x` and `y` directions (which I `OR` together) then this result is `AND`-ed with a magnitude threshold. The colour and gradient thresholds are then `OR`-ed together.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In the `image_correction.py` file there is a function `warp_perspective` that transforms the image into the "birdseye" perspective. I found it was necessary to hard code the `src` and `dst` points for each specific image source (i.e. each of the videos). I guess this is due to the relative position of the road to the camera due to hills etc. For example my `src` and `dst` points for the sample video were

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 493, 541      | 100, 1100     |
| 857, 541      | `image.shape[0]`-100, 1100      |
| 347, 665      | 100, `image.shape[1]`      |
| 1102,665      | `image.shape[0]`-100, `image.shape[1]`        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To detect lane lines I used th sliding window approach. The two window that had the highest number of hot pixels (either side of the center) were said to contain the lane line. The centers of the slected windows were then used to fit a second order polynomial. The code for this can be found in `fit_lines` in the `detect_lanes.py` file.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Built in to the `Line` class is a function (`calculate_curvature`) that is used to calculate the curvature of the line. Using simple calculus we are able to find the radius of curvature based on some of the derivative properties of the fitted line. There is another function inside the `Line` class (`get_line_base_pos`) that will calculate the position of the vehicle based on the two fitted lines (so this means you also need to pass the fit of the other line to the class). All it does is find the average position of the two lines at the bottom of the screen and finds the difference between this and the center (we have assumed that the camera is mounted on the center of the car).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In the pipeline this drawing and calculation step is handled by the `visualise_lanes` function in the `detect_lanes.py` function.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/EhfuW00c08E)

I also had a go at the [challenge_video.mp4](https://youtu.be/EZ9JD-ql8Yc) and [harder\_challenge\_video.mp4](harder_challenge_animation.mp4) videos too.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I don't know why but my biggest problem seemed to be the warping of the image. It just seems so manual and I feel as though there should be a better way to do it. I found that I need to recalibrate my warping for each test video, which doesn't really make sense given that all videos could, in theory, have been shot in a single trip. This would mean that a car using my algorithms would fail the moment the slope in the road changed or something bumped the camera.

Determining the thresholds for the different colour channels and gradients was pretty straight forward. Figuring out the best way to combine the results from the different kinds of thresholding was interesting both from a logical perspective as well as a practical one.

Even though I have used quite a few different thresholds my lane detection still doesn't appear to be as robust as it could be. It struggles a little bit with the shadow towards the end of the project video when it really should have enough to handle it well. So I am not entriely sure whta went wrong there.

I didn't spend any time optimising the lane finding algorithms. So each frame is built from a full window scan, instead of using previous knowledge from the past frame. This may help in reducing the spurious flickering that can be observed in some of the difficult parts of the videos.
