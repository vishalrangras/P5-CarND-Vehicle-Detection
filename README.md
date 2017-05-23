# Vehicle Detection Project

**Submitted by: Vishal Rangras**

## Attributions

1. I would like to give credits to Rana Khalil's work which gave me a good understanding about importance of overlapping windows and window size. I have experimented with different values for these two parameters and ultimately used the values similar to Rana Khalil's work for my final video processing.

2. I would also like to give credits to forum mentor Subodh Malgonde for helping me to resolve the black screen output in final processed Video. As it turned out, my processing pipeline was processing the image in the range of 0 - 1 due to which the final output was of complete black screen with only boxes in the location of car. I was not able to figure out the solution to this as my pipeline was working fine on single still image but was not producing correct output for video. So I asked the question on Discussion Forum and slack channel. Ultimately Forum Mentor Subodh Malgonde recommended me to have a copy of original image and to do all the processing on the copy, while using the original image for final plotting of results. This ensured the satisfactory result.

3. I was also having hard time understanding the concept of HOG and what does each parameter means in hog() function. Forum mentor Francesco_F helped me understand this concepts by providing a link to blog: https://www.learnopencv.com/histogram-of-oriented-gradients/
This blog helped me understand the concepts of HOG and its parameter.

I am thankful to above people / resources for enabling me to get proper understanding of important concepts and helping me in resolving the issues faced by me.

4. Last but not the least, almost all the code which I have written in the project is taken from Udacity's Classroom content and I have just tweaked it a little or experimented with parameters to achieve the final output so the Udacity also deserves the credit for this project work.

---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./test_images/test4.jpg
[image2]: ./final_window_img.jpg
[image3]: ./final_draw_img.jpg
[image4]: ./output_images/data_look.jpg
[image5]: ./output_images/spatial_binning.jpg
[image6]: ./output_images/YCrCb_color_hist.jpg
[image7]: ./output_images/HOG_visualization.jpg
[image8]: ./HOG_Features/010-All_Channels_of_YCrCb.JPG
[image9]: ./HOG_Features/004-All_Channels_of_HSV.JPG
[image10]: ./Sliding_Experiment/window_img_64_0.8_SVM_Change_3_F.jpg
[image11]: ./Sliding_Experiment/window_img_96_0.9.jpg
[image12]: ./output_images/HSV_color_hist.jpg
[image13]: ./output_images/raw_and_normalized_features.jpg

[video1]: ./project_video_processed.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Before we get started about HOG and other features of images, let us have a look at the training dataset which we are using for this project. The dataset consist of vehicle and non-vehicle images and a sample of this is shown below:

![alt text][image4]

The function to compute HOG i.e. `get_hog_features()` is written in Cell 4 of my Jupyter Notebook along with other functions like `bin_spatial()` and `color_hist()`. Its visualization is displayed in Cell 7 of the notebook.  
The HOG feature is using following parameters:
| | |
| orient | 9 |
| pix_per_cell | 8 |
| cell_per_block | 2 |

I have also used Spatial binning and Color Histogram along with HOG for feature extraction. The following figures illustrates the various features extracted from the input images:

![alt text][image5]

![alt text][image6]

![alt text][image12]

![alt text][image7]

![alt text][image13]

In earlier iterations of my project, being lazy I just used KITTI dataset for the training purposes and feature extraction. Classifier trained only on KITTI dataset worked fine on single image. However, the output I received due to this was such that only back portion of the car was detected but the side portions were not detected by my classifier. I believe the reason behind this is because KITTI dataset contains all the images from the single perspective angle i.e. back.

Then I used all the available images of GTI dataset along with KITTI dataset which lead to a better accuracy and improved vehicle detection.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various values for HOG parameters and all the changes directly impacted the feature vector length along with change in time required to train SVM Classifier and its accuracy. All the results are placed in **"HOG_Features"** directory for the reference. Here are two different value sets shown for among the all tested values:

![alt text][image8]

![alt text][image9]

I was able to reduce the training time by increasing value of orient and reducing hist_bin and spatial_size values. This also showed a good training accuracy but the classifier was not giving satisfying results and finally I ended up using the HOG parameters similar to Udacity's Classroom content itself.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Cell 9 and 10 contains the parameters which can be tweaked before extracting features and a call to extract car and non-car features. These features are then used for the purposes of training Linear SVM in cell 12. Like I mentioned above, I tried various combinations of feature parameters which lead to different feature vector every time. Finally I settled with the parameters similar to Udacity and my feature vector was having the length of 8460.

My SVM classifier got trained in 217.8 seconds on a normal I3 processor with 8 GB RAM. I really liked and enjoyed this as for deep learning, I could never train any data on my local but I was able to do so with Machine learning as it does not demand as much compute as Deep Learning algorithms. The SVM classifier had a test accuracy of 0.9899 which I believe is good accuracy.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code related to Sliding Window search is located in Cell 13 designated as **"Car Search"**. The parameters for sliding window size and overlap are located in Cell 14 along with the code to actually search the cars in an image and draw the boxes on the image as per the search result.

I tried so many different values of xy_window and xy_overlap along with different 3 different feature vectors. The various results are stored in **Sliding_Experiment** directory. Finally I got the best result from Rana Khalil's work for these parameters like I mentioned above in the attribution. The final values which I used are: xy_overlap = (0.8, 0.8) and xy_window = (80,80)

Here are two sets of values for `Sliding Experiment` among all the values:

![alt text][image10]

![alt text][image11]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I have mentioned above all the iterations I tried to finally reach the desired results but to sum them up, I did the 3 major changes as follow:

1. Increased the training dataset by considering GTI images along with KITTI images. My 1st iteration only had KITTI images in it and GTI were skipped.
2. Earlier iterations has parameters of feature extraction such that the SVM was getting trained very quickly due to small feature vector length and was still providing correct accuracy on training/test data. However it was not working well on video pipeline so the parameters were changed to original Udacity parameters.
3. Experimented with Window size and window overall to get the best possible vehicle detection avoiding false positives.

Here are the images showing working of pipeline.

![alt text][image1]
![alt text][image2]
![alt text][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_processed.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As taught in the classroom lectures, heatmap and thresholding was used to sum up the nearby frames detected as vehicle. Then label() function was used to identify vehicles in the image. The functions for this are defined in cell 15 and been called in cell 16 and processing pipeline.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Like I mentioned above, I was was first facing the issue that my pipeline was detecting only back portion of cars and not the side portion. This was resolved by adding more training data. Then I faced the issue of low detection area of car which was fixed by manipulating sliding window search parameters. And finally I faced the issue of having a video outputting complete black screen with only bounding boxes in it describing the location of car in the frame. This was fixed by maintaining a copy of original frame in the processing pipeline and then using this copy to draw bounding boxes. The issue occurred due to range conversion of (0 - 1) and (0 - 255).

I believe that this pipeline will likely fail for any new type of car it sees as it heavily depends on the models of cars and their specific shapes. It cannot do a job similar to human eye by detecting any new type of car with a different shape then usual car shapes. Also, I believe that my pipeline is very slow, at least it took a lot of time on my slow system and it can be optimized to become more faster and accurate relatively.

One funny scenario which comes to my mind where this pipeline might fail is that in India, sometimes people keep the trunk of the car open and they load some stuff in it. If such vehicle is going on the road, this pipeline might not detect it as a car I believe.