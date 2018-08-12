# **Finding Lane Lines on the Road in Image and Video data Writeup** 

## Objectives

The main objective is to create a software pipeline with certain helper functions to detect lane lines in image and video data. The following methods were implemented to obtain the final results:

1. Color selection to detect white and yellow lane lines
2. Convert image to grayscale for Canny Edge detection
3. Gaussian Blur to remove noise from the image
4. Canny Edge detection
5. Define a Region of Interest
6. Hough Transform
7. Average and extrapolate hough lines
8. Draw annotated lines on image and video data
9. Main pipeline

## Reflection

### 1. Software Pipeline
---
The flow of the software piepline is explained in the following sections along with the detailed explaination of each helper function and the reasons for selecting various parameters in them:

  ### 1. Color selection to detect white and yellow lane lines
  After reading an image, the first task is to robustly detect the white and yellow lines in an image. 
  ##### Original image
  ![challenge_image1](https://user-images.githubusercontent.com/8627486/43037545-1e742d10-8cdc-11e8-9efd-748017139e6c.png)
  
  If I convert this image to grayscale and use the ```inRange()``` function with parameters set to detect yellow and white lines, the output is not very clear in situations where shadows are present on the lane lines.
  ##### Grayscale image
  ![grayout](https://user-images.githubusercontent.com/8627486/43037471-47413482-8cdb-11e8-9757-5f26069dd0b3.png)
  
 I converted the original image to a HSV image and observed that the color selected output performs better than a color selected grayscale image. Although, in comparison with the above two formats, the color selected output of a HSL image was much better in detecting white and yellow lane lines in image and video data without getting affected by any shadow on the road. Hence, I perform the further image processing techniques on a HSL image.
 ##### HSL image
 ![hslout](https://user-images.githubusercontent.com/8627486/43037612-fc394eaa-8cdc-11e8-95c9-fc3ff8ec46a5.png)
 
 To detect white lines, I selected a high range of Light values (Range 190-255). To detect yellow lines, I selected a lower range of Hue values (Range 0-150) and a wider range of Light values (Range 100-255). The final output of a color selected HSL image looks as follows:
 ##### Color selected HSL image output
![hslfilout](https://user-images.githubusercontent.com/8627486/43037702-622aa64a-8cde-11e8-9ae5-baf95dd78378.png)
 
 ### 2. Convert image to grayscale for Canny Edge detection
 As the canny edge detector requires a grayscale image (as the algorithm looks for gradient values) as an input image, we convert the color selected HSL image into a grayscale image and give it to the canny edge detector.
 ##### Color selected HSL image converted to grayscale
 ![grayfilout](https://user-images.githubusercontent.com/8627486/43037727-ec5a12ec-8cde-11e8-8e32-854679044456.png)
 
 ### 3. Gaussian Blur to remove noise from the image
 To smoothen the grayscale color selected HSL image, we apply a Gaussian Blur with ```kernel_size = 7``` and thresholds as ```low_threshold = 50```, ```high_threshold = 150```.
 ##### Applying Gaussian Blur
 ![blurout](https://user-images.githubusercontent.com/8627486/43038204-2f666afa-8ce3-11e8-8ff4-14d5cd15bde6.png)
 
 ### 4. Canny Edge detection
 Apply Canny Edge detector to the smoothen image to obtain an image with all the edges detected.
 ##### Applying Canny Edge detector
 ![edgeout](https://user-images.githubusercontent.com/8627486/43038223-6b31d6c8-8ce3-11e8-82da-cfbfd7a04400.png)
 
 ### 5. Define a Region of Interest (ROI)
 Define a polygon to exclude all the irrelevant edges in the image so that we can only see the lane lines on the road.
 ##### ROI output
 ![roiout](https://user-images.githubusercontent.com/8627486/43038250-efd78e68-8ce3-11e8-8b33-dd4f8b751832.png)
 
 ### 6. Hough Transform
 Find hough lines in the ROI output image using hough transform
 ##### Hough Transform output
 ![houghout](https://user-images.githubusercontent.com/8627486/43038461-eb6d62fa-8ce6-11e8-8a99-9ac5566966d6.png)
 
 ### 7. Average and extrapolate hough lines
   * Average function - ```average(lines)``` 
   
   First I created two lists ```leftLine``` and ```rightLine``` which would store the weighted average of all the hough lines      detected in the left and right lane respectively. To classify if a hough line belongs to the left or right lane, we            calculate the slopes of all hough lines (intercepts and length as well). Hough lines with negative slope and length more than    50  pixels get classified as left lane lines and hough lines with positive slope and length more than 50 pixels get              classified as right lane lines. Taking a weighted average of all the classified left and right lane hough lines based on the    length of the lines help eliminate all the smaller hough lines detected that can make the final output unstable. Hence, only    the longer hough lines detected will dominate the weighted average so that the output is robust.
   
   * Find Endpoints function - ```findEndpoints(image, intercepts, yLimit)```
   
   I created two lists namely ```global prev_leftLaneLine``` and ```global prev_rightLaneLine```. These lists will store the ```leftLine``` and ```rightLine``` values of the previous frame while comparing the corresponding values in the current frame only when a NONE case is observed in the ```leftLine``` and ```rightLine``` values. The reason for getting a NONE case is that may be for a certain frame, all the hough lines detected have only positive or only negative slope. That means no hough lines were classified in either the ```leftLine``` and ```rightLine``` lists resulting in a empty list or a NONE case. The pipeline crashes if this NONE case is not handled. Hence, if a NONE case occurs, using the  ```global prev_leftLaneLine``` and ```global prev_rightLaneLine``` values, I find the endpoints of the final left lane and right lane to be drawn on the image. If no NONE case occurs, simply use the current frame's ```leftLine``` and ```rightLine``` values to calculate the final left lane and right lane to be drawn on the image. ```yLimit`` is the top endpoint of ROI on the Y axis.

### 8. Draw annotated lines on image and video data
   * Draw Lane Lines function - ```drawLaneLines(image, endpoints, color=[[255, 0, 0], [0, 0, 255]], thickness = 15)```
   
   If there is no NONE case from the previous function, I use the ```cv2.line()``` function to draw an annotated left lane on the image in RED color and an annotated right lane on the image in BLUE color. I also use the ```cv2.fillPoly()``` function to fill the area between the left and right lane in GREEN color to represent "SAFE DRIVING ZONE". I also add weights on the final image using ```weightedImage(image, initial_image, α=0.5, β=1., γ=0.)```.

 ##### Applying Averaging and extrapolating functions
 ![avgout](https://user-images.githubusercontent.com/8627486/43038651-4d6d0d4e-8ceb-11e8-8e34-59bf3caf6ffe.png)
 
 ### 9. Main pipeline
 Using all the above helper functions, when I first tested my pipeline on the video data, everything worked fine but I observed a lot of jitter in the lane lines holding their position (bouncing around slightly) along the entire length of the video. The pipeline worked but it wasn't robust enough ! After researching online, I found out the reason for this jitter to be slight deviations in the lane lines values from frame to frame which causes this jittering effect. Hence, I stored the lane line outputs of the most recent 50 frames in ```prev_LaneLines[]``` and took an average of the most recent 50 frames output with the current frame's lane line output to display the current frame's output. I made a class ```class DetectLanes``` and embedded my pipeline inside it so that I can use an object of this class ```detect = DetectLanes()``` to display the mean output of my pipeline on image and video data. NOTE: I had to flush the ```prev_LaneLines[]``` array using ```del detect.prev_LaneLines[:]``` before running the next video file as it used the data from the previous video file's execution.
   
 # Software Pipeline Output
 [Images](https://github.com/AllenMendes/Finding-Lane-Lines-in-Image-and-Video/tree/master/CarND-LaneLines-P1/test_images_output)
 ---
 [Videos](https://github.com/AllenMendes/Finding-Lane-Lines-in-Image-and-Video/tree/master/CarND-LaneLines-P1/test_videos_output)
---
 [Extra Video](https://github.com/AllenMendes/Finding-Lane-Lines-in-Image-and-Video/tree/master/CarND-LaneLines-P1/test_videos_output_extra)
---

### 2. Potential shortcomings of current pipeline
The current software pipeline will not work:
1. If there are curved roads (mountain roads)
2. If there are too many shadows on the roads
3. If there no lane markings or lane markings are other than yellow and white color lines
4. If the image/video perspective is not head on (horizon level changes)
5. Drastic changes in ligthing conditions (rain, night time, snow)
6. Cars or external objects blocking the region of interest or lane lines too much

### 3. Possible improvements of current pipeline
1. Add perspective transform
2. Making pipeline independent of light variations and shadows
3. Adding more sensor data like LIDAR and GPS in addition to the visual data


