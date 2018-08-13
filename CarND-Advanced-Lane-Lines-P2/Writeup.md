# **Advanced Lane Finding on a Road in Image and Video data Writeup** 

## Objectives

The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
2. Apply a distortion correction to raw images
3. Use color transforms, gradients, etc., to create a thresholded binary image
4. Apply a perspective transform to rectify binary image ("birds-eye view")
5. Detect lane pixels and fit to find the lane boundary
6. Determine the curvature of the lane and vehicle position with respect to center
7. Warp the detected lane boundaries back onto the original image
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
9. Complete pipeline

## Reflection

### 1. Software Pipeline
---
The flow of the software piepline is explained in the following sections along with the detailed explaination of each helper function and the reasons for selecting various parameters in them:

  ### 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images 
  (__*Code Section- Camera calibration using chessboard images*__): I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, ```objp``` is just a replicated array of coordinates, and ```objpoints``` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. ```imgpoints``` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.
  
  Once the corners are detected, I draw these corners on the [camera calibration images](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/tree/master/CarND-Advanced-Lane-Lines-P2/camera_cal) using ```cv2.drawChessboardCorners()``` and display all the images on which all the corners were detected or not. 
  #### Camera Calibration
  ![cam_cal1](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/output_images/Camera_Calibration1.jpg)
  ![cam_cal2](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/output_images/Camera_Calibration2.jpg)
  
  ### 2. Apply a distortion correction to raw images
  (__*Code Section- Undistort image*__): I then used the output ```objpoints``` and ```imgpoints``` to compute the camera calibration and distortion coefficients using the ```cv2.calibrateCamera()``` function. I applied this distortion correction to a chessboard image and test image using the ```cv2.undistort()``` function and obtained this result: 
  #### Undistort image
  ![chess_test](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/output_images/undistortChessboard.jpg)
  ![undistort_test](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/output_images/undistort_test5.jpg)
  
  ### 3. Use color transforms, gradients, etc., to create a thresholded binary image
  (__*Code Section- Color and Gradient thresholding*__): I used the L channel to detect white lanes and the S channel to detect yellow lanes with certain thresholding limits on the HLS values. As we are only interested in vertical lane lines with respect to the vehicle, I used the Sobel operator/filter to take a derivative (gradient descent) only in the X direction. Hence I obtained the gradient descent of the entire image only in the X direction with certain thresholding limits. On combining the color and gradient thresholded binary images, following is the binary output:
  #### Color and Gradient Thresholded image
  ![thres-out](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/output_images/thres_out.jpg)
 
 ### 4. Apply a perspective transform to rectify binary image ("birds-eye view")
 (__*Code Section- Perspective Transform*__): I hardcored the source and destination points to extract a part of my binary output image (Region of Interest - ROI) and convert it to a bird's eye view for further calculations. The source and destination points are as follows:
 
 | Location | Source   | Destination |
 |:--------:|:--------:|:-----------:|
 | Top Left | 450, 550 |    450, 0   |
 | Top Right | 830, 550 | 830, 0 |
 | Bottom Left | 230, 700 | 450, 720 |
 | Bottom Right | 1075, 700 | 830, 720 |
 
 Using the above values, the perspective transformed binary image looks like follows:
 #### Perspective Transformed image
 ![pers-out](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/output_images/persTrans_out.jpg)
 
 ### 5. Detect lane pixels and fit to find the lane boundary
 (__*Code Section- Pipeline*__): Using all the above functions, I created a software pipeline to find perspective transformed binary images of all the [test images](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/tree/master/CarND-Advanced-Lane-Lines-P2/test_images). 
 #### Pipeline Output
 ![pipe-out](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/output_images/pipeline_out.jpg)
 
 (__*Code Section- Sliding Window Polyfit*__): This function uses a sliding windows to fit a second order polynomial on the lane lines detected in the perspective transformed image. First, I start off by taking a histogram of all the pixels in the bottom half of the image. I narrow down the region for finding lane lines to the quarter of the histogram on either sides of the midpoint of the image. Starting from the left and right base points of the left and right detected lane lines, I create a rectangular window and determine all the non zero indices and append them into two arrays. I shift the windows upwards by the height of the window and repeat the above mentioned step. In this way, I obtain all the indicies of the left and right lanes which are detected. Using ```np.polyfit()```, I can fit a second order polynomial based of on the left and right lanes indices. The output looks like this:
#### Sliding Window Polyfit Output
 ![slide-out](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/output_images/slidingWindow_out.jpg)
 
 (__*Code Section- Fit polynomial based on previous frame's polyfit*__): We don't need to repeat the entire sliding window polyfit routine for every frame. Instead, if the previous frame correctly detected the lane lines, we can continue fitting the polynomial based on previous frame's data by this function. 
 
 
 ### 6. Determine the curvature of the lane and vehicle position with respect to center
 (__*Code Section- Radius of Curvature and Distance from Lane Center Calculation*__): To determine the radius of curvature and distance of the vehicle from lane center, I first need to convert the pixel information about lanes to meters. The actual lane width is 12 ft (3.7 meters) and the length of a dashed lane line is 10 ft (3.048 meters). The distance between ```left_fit_x_int``` and   ```right_fit_x_int``` (line 121) or lane width is 380 pixels and the length of a dashed lane line (```righty```-line 87) is 430 pixels. Hence, lane width in meters would be **3.7/380** and length of dashed lane line in meters would be **3.048/430**. Radius of curvature is obtained by (lines 31-32):
 ![rad](https://user-images.githubusercontent.com/8627486/44008365-ec1a4efc-9e70-11e8-8a93-15acb1de0429.JPG)
 
 The left lane and right lane radius of curvature is obtained at the base of the image and the average of the left and right lane radii is the radius of curvature of the lane. The current base points of the left and right lane is used to determine the lane center. As the vehicle is considered to be at the center of the image, we can easily find the lane center offset (line 41) in meters.
 
 ### 7. Warp the detected lane boundaries back onto the original image
 (__*Code Section- Draw detected polyfitted lanes back onto the original image*__): Using ```cv2.polylines()``` and ```cv2.fillPoly()``` I can annote the polyfitted lane lane from the binary image. The inverse of the perspective transform matrix (Minv) is used to wrap the bird's eye view image onto the original image. The output looks like follows:
 #### Drawing lanes line
 ![draw-out](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/output_images/drawLanes_out.jpg)
 
### 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
 (__*Code Section- Draw curvature radius and distance from center data onto the original image*__): Determine the lane center offset direction i.e. left or right of the vehicle center and display all the information as follows:
 #### Final Output
 ![final-out](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/output_images/final_out.jpg)
 
 ### 9. Complete pipeline
 (__*Code Section- Define a Line Class for Storing Data*__): In this function, we create variables to store useful information like is line detected or not, current fit, best fit, etc. The ```addFit()``` function will add the current fitted polynomial into an array only if the difference of the current fit and the best fit found so far is greather than a emperical threshold. Only the most recent ```prevLinesCount``` fits will be stored in the this array and the average of the array along with the current fit will be used to display the results. If a good fit is not detected in a current frame, then the oldest one fit will be taken off from the array.
 
 (__*Code Section- Define Complete Image Processing Pipeline*__): If left and right lanes are detected in current frame, use ```slidingWindowPolyfit()``` function else use previously detected lane lines using ```polyfitPrev()```. If the distance between the two lanes is too much (wrong detection), then invalidate those fits. In the end, display the results as the average of the current fits and the best fits detected.
   
 # Software Pipeline Output
 [Images](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/tree/master/CarND-Advanced-Lane-Lines-P2/output_images/test_images_output)
 ---
 [Project Video](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/test_videos_output/project_video_out.mp4)
 
 [Challenge Video](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/test_videos_output/challenge_video_out.mp4)
 
 [Harder Challenge Video](https://github.com/AllenMendes/Advanced-Lane-Finding-in-Image-and-Video-data/blob/master/CarND-Advanced-Lane-Lines-P2/test_videos_output/harder_challenge_video_out.mp4)
---

### 2. Potential shortcomings of current pipeline
The current software pipeline will not work:
1. If the roads have too much curves and U turns (mountain roads)
2. If there are too many shadows or bright spots on the roads
3. If there no lane markings or lane markings are other than yellow and white color lines
5. Drastic changes in ligthing conditions (rain, night time, snow)
6. If cars crossover the lane lines resulting in a wrong detection

### 3. Possible improvements of current pipeline
1. Make the pipeline data driven (Train a CNN over image and video data) instead of rule/logic based 
2. Making pipeline independent of light variations and shadows
3. Adding more sensor data like LIDAR and GPS in addition to the visual data
