## P5 Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./forreport/car_notcar.png
[image2]: ./forreport/hogcar.png
[image3]: ./forreport/hogcar2.png
[image4]: ./forreport/hogcar3.png
[image5]: ./forreport/hognotcar1.png
[image6]: ./forreport/hognotcar2.png
[image7]: ./forreport/hognotcar3.png
[image8]: ./forreport/note.png
[image9]: ./forreport/slidingwindow1.png
[image10]: ./forreport/test10sw.png
[image11]: ./forreport/test9sw.png
[image12]: ./forreport/test7sw.png
[image13]: ./forreport/test10heat.png
[image14]: ./forreport/test9heat.png
[image15]: ./forreport/heatmap.png
[image16]: ./forreport/test2Heat.png
[image17]: ./forreport/test7heat.png
[image18]: ./forreport/test10lb.png
[image19]: ./forreport/test9lb.png
[image20]: ./forreport/test8lb.png
[image21]: ./forreport/test2lb.png
[image22]: ./forreport/test7lb.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the **fourth** code cell of the IPython notebook .  



I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

**Car**

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

#### 2. Explain how you settled on your final choice of HOG parameters.

Codes for this section is called **Experiment : Chooseing HOG Perimeter** in ipython notebook.

I first extract a random sample consisting 1000 files from the `car` and `notcar` dataset for reducing experiment time.

Then I set several (3 or 4) candidate values for each parameter, such as `[6, 9, 12]` for `orientation`, then use `for` loop iterating each combination of parameters.   It runs 4 (color space) * 3 (orientation) * 2(pix_per_cell) * 3 (cell_block) * 4 (color channels) = 288 times . After reviewing these test score, I got the insights that `'ALL'` color channel is better than single channels, `'YCrCb'`and ` 'HSV'` perform better than other color space most of time.

Meantime, from the lesson's intructor's notes, the lecture shows some best practice:

![][image8]

I then narrow the search space, only provide 2 candidate value for each parameter. The final results shows that the combination of 

```
# Best parameter
color_space = 'YCrCb'
orient = 18
pix_per_cell = 12
cell_block = 1
hog_channel = 'ALL'
```

performs best and got 0.99 score on the test dataset.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code is entitled "**SVC Model Traing**" in the notebook.

I first gather all cars and notcars image from the dataset (at the begining of the notebook), then extract hog feature from these raw images. I use `StandardScalor()` from `sklearn.preprocessing` to normalize and reduce variance of the feature . After that I use `train_test_split()` with `0.2` ration of test set  to split the whole dataset into training and testing data. 

Then I import `LinearSVC()`model from `sklearn`to train on the HOG features, and use test data for validation. The final score is `0.9789`

Finally, I store the svc model into pickle.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is in **7.Sliding Window**. I first use `slide_window()`to collect window for one size. then define `multi_scale_window()` for generating multiple scale windows.

I decided to search the space in 4 scale: (64,64) for distant small object, (96, 96) for moderate distance object, large and extreme large for closest object. 

the shape of cars on your right or left is a rectangle, so I set extreme large box ratio of width/height to 4:3.

|         | size    | y_start | y_stop | x_start | x_stop | overlap |
| ------- | ------- | ------- | ------ | ------- | ------ | ------- |
| small   | 64*64   | 360     | 576    | 256     | 1280   | 0.5,0.5 |
| medium  | 96*96   | 360     | 576    | 256     | 1280   | 0.8,0.8 |
| large   | 128*128 | 360     | 684    | 256     | 1280   | 0.6,0.6 |
| extreme | 256*192 | 360     | 684    | 256     | 1280   | 0.6,0.6 |


![alt text][image9]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YCrCb 3-channel HOG features , which provided a nice result.  Here are some example images:

![alt text][image10]

![alt text][image11]

![alt text][image12]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_out.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code is in **8. Heat map** 

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

#### Here are some frames and their corresponding heatmaps:

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]

![alt text][image17]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image18]

![alt text][image19]

![alt text][image20]

![alt text][image21]

![alt text][image22]



---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think that the biggest drawback of sliding window is that you have to pre-defined the window size and overlap rate, which may not generalize well. The pipeline in my implementation only process the currently frame thus sometimes fail to track the object all the way. And Heat map is a great way for eliminating false positives but may omit small or distant objects for lacking bounding boxes. 

Here are 3 ideas for future improvements:

+ Use selective search gererating window automatically rather than sliding window approach.
+ Use NMS to reduce box overlapping 
+ Use HOG sub-sampling to reduce processing time



