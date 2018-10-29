# **Behavioral Cloning**

## Writeup

### I use this markdown file as a summary report for my Behavioral Cloning Project.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nVIDIA_CNN_model.JPG "Model Visualization"
[image2]: ./examples/training_dataset_1.JPG "Training Dataset from Udacity"
[image3]: ./examples/training_dataset_2.JPG "Training Dataset after combining right/left/center images"
[image4]: ./examples/training_dataset_3.JPG "Final training images"
[image5]: ./examples/image_crop.JPG "Cropped Image"
[image6]: ./examples/image_resize.JPG "Resized Image"
[image7]: ./examples/image_flip.JPG "Flipped Image"
[image8]: ./examples/image_random_brightness.JPG "Image Random Brightness"
[image9]: ./examples/image_gaussian_blur.JPG "Image Gaussian Blur"
[image10]: ./examples/image_trans.JPG "Image Translation"
[image11]: ./examples/image_warp.JPG "Image Warp"
[image12]: ./examples/model_layers.JPG "model layers"
[image13]: ./examples/running_result.JPG "running result"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model_jupyter.ipynb - I used Jupyter Notebook to develop and debug the model before I convert it to the above model.py
* drive.py for driving the car in autonomous mode
* model.h5 (also model_02.h5, model_05.h5 and model_06.h5) containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py or model_jupyter.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I implemented for the project is [nVIDIA model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
suggested by the Udacity's course. The graph below is nVIDIA's self-driving car model. The nVIDIA's model includes 9 layers - 1 normalization layer, 5 convolutional layers and 3 fully connected layers.

![alt text][image1]

I reproduced the nVIDIA's model by Keras Sequential model. The model starts with a Lambda function to normalize the input data(images).  Next, five convolutional layers are designed to extract features. The first three convolutional layers have a 2x2 stride and 5x5 kernel. The last two convolutional layers have a 3x3 kernel without stride. Those five convolutional layers adopt `valid` padding method and `l2` kernel regularizer. Exponential Linear Unit (ELU) activation function is followed after each convolutional layer. Then three fully connected layers are designed as output function to control the vehicle wheel steering.

#### 2. Attempts to reduce overfitting in the model

The convolution layers includes L2 kernel_regularizer with scale 0.001. The model also contains dropout layers in order to reduce overfitting. A dropout function is followed after each fully connected layer. The dropout function is implemented after three fully connected layers. The probability for the dropout function is 0.5.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The `Adam` (`ADA`ptive `M`oment estimation) optimizer is chosen with a learning rate 0.001. The loss function is mean squared error (`MSE`).

#### 4. Appropriate training data

#####  step 4a)
Udacity provides a dataset. All my training dataset is based on Udacity's data. The steering angle histogram is shown as below:

![alt text][image2]

20% of the above training dataset is used for validation and the remaining 80% is further preprocessed for training dataset.

I used a combination of center lane, right and left lane images. When I appended left lane images to the training dataset, I applied an steering angle offset 0.25. Same as to the right lane images, an angle offset -0.25 is applied. the combine function for the training dataset is `training_data_altogether(center, right, left, steering_measurements)`.

Below is a distribution of steering angle after combining left, right and center lane images.

![alt text][image3]


#####  step 4b)
Inspired by [Shannon's post](http://jeremyshannon.com/2017/02/10/udacity-sdcnd-behavioral-cloning.html), I made histogram graph for the steering angle with 23 bins. Then an average sample number is calculated based on 23 bins. I decided if the sample number from one bin is greater than twice of the average sample number, the extra images will be discarded randomly.

#####  step 4c)
Next, I classified the training dataset with straight turn, right turn and left turn based on `steering_threshold (0.25)`. The function is `right_left_straight_SteeringClassifier(center, steering_measurements, steering_threshold)`.

Then I also artificially created recovery data. For left turn, when a steering angle is less than `-steering_threshold (-0.25 degree)`, I appended its right lane image to the training set with an steering angle offset `steering_adjustment (0.25)`. Same as for the right turn,  when an steering angle is greater than `+steering_threshold (+0.25 degree)`, I appended its left lane image to the training set with an steering angle offset `steering_adjustment (0.25)`. Below is the histogram of the final train dataset for the model input after the above step 4b and 4c.

![alt text][image4]

#### 5. Data Loading by Implementing a Python Generator in Keras
Because we have much larger image data than that in the previous project `Traffic Sign Classifier Project`, __Generator__ is introduced for the data loading and augmentation. Generator is a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator we can pull pieces of the data and process them on the fly only when we need them, which is much more memory-efficient. I created 2 generators for the project. One is for the training dataset - `generator_data_train(batch_size, X, y)`. The other is for the validation dataset - `generator_data_valid(batch_size, X, y)`.



#### 6. Data Augmentation
Several data augmentation methods have been implemented after data is loaded into the __generator__ for the model. Some of the augmentation methods are adopted after reading [David Silver's post](https://medium.com/udacity/how-udacitys-self-driving-car-students-approach-behavioral-cloning-5ffbfd2979e5).
Those methods are listed below:

##### 6a) Image Crop
The original image shape from the cameras in the simulator is 160 pixel by 320 pixel. Not all of these pixels contain useful information, however. In the image below, the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. In the function `image_crop(image)`, 60 rows pixels from the top of the image, 20 rows pixels from the bottom of the image are removed. The graph below is a comparison before and after the function `image_crop(image)` is applied.

![alt text][image5]

##### 6b) Image Resize
The nVIDIA's model requires an input shape 66x200, the function `image_resize(image)` is to resize an image shape to 66x200. The graph below is illustrated on how `image_resize(image)` function works.   

![alt text][image6]

##### 6c) Image Flip
Most of the data from the track 1 are left turns. An effective technique for helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement. The function `image_flip(image, angle)` is developed to flip the image. The graph below shows a comparison before and after image flipped.

![alt text][image7]

##### 6d) Image Random Brightness
A function of random brightness is developed `random_brightness(image)`. It converts an image from BGR to HSV and applies a random factor to the V channel. Then it's converted back to the BGR image. Then purpose here is to randomly create some darker light images.   
![alt text][image8]


##### 6e) Image Gaussian Blur

Gaussian blur function is applied to smooth the image. The kernel size is 3x3. You can find it `gaussian_blur(image)`
Below is a graph showing how `gaussian_blur(image)` works.

![alt text][image9]

##### 6f) Image Translation

A function of `trans_image(image,angle)` is developed to horizontally and/or vertically move the image. For the final result, I only moved the training images horizontally. A factor of 0.008 steering angle per pixel compensation is applied in the function. Below is an example of the function output.    

![alt text][image10]


##### 6g) Image Warp

A function of `image_warp(image, angle)` is developed to warp the image horizontally. A factor of 0.004 steering angle per pixel is applied in the function. Below is an example of the function output.    

![alt text][image11]



#### 7. Model Tuning and Final Results
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The above data augmentation methods are applied in the training and validation Generators. For the training generator, I applied all the above data augmentation methods. Some methods are applied randomly. For example, `random_brightness(image)`, `trans_image(image,angle)`, `image_warp(image, angle)`, `image_flip(image, angle)`. The other methods are applied for all the images both in training and validation generators. Those are `image_crop(image)`, `image_resize(image)`, `gaussian_blur(image)`.  

Overfitting is a big challenge. In order to reduce overfitting, I tried different dropout keep probability for each fully connected layers. After several tryouts, I ended up with 0.5 for keep probability. For the convolution layers, I applied 0.001 L2 kernel regularizer. I didn't get many chances to test other values.

For the activation function, after reading some posts online, I ended up with Exponential Linear Unit (ELU) activation function. In the future, I could try other activation functions to see if the model can get improvements.

The Adam optimizer, for the submitted results, the learning rate is 0.001. I could also try other values to see if it helps.

Below is a model summary for the submitted result. I ran the model on my own laptop with CPU. (I don't have a nVIDIA GPU). As you can see, each epoch took about 1 hour and 30 minutes. In the future, I could run it in AWS or other GPU capable computers.

![alt text][image12]

One useful tool in Keras framework is the callback feature. With that, at the end of each epoch, the model weights could be saved. As some online posts already pointed out, in this project, the loss was not an entirely reliable indicator for the model performance. That's why it's import to save training weights for each epoch. In my case, the submitted result, I ran 6 epochs. I found the validation loss actually increased. I tested all 6 model weights, the results from epoch 2, epoch 5 and epoch 6 can be used to successfully operate the vehicle drive autonomously around the track without leaving the road.

![alt text][image13]


#### 8.  Conclusion and Discussion

From this project, I find it's a tough challenge to tune the hyperparameters for the model. You can get some senses which tuning could make the model go to the right direction. But it's hard for me to quantify the improvements with my limited time and limited hardware. I spent 2 weekends to get the model running and optimizer the model.

Other than the model tuning, the more important thing for the project is the data and data augmentation. As you can see, the steering angle distribution from the training dataset is not well distributed. Most of them are from the straight lane with zero or near zero values. It would definitely cause a large bias for the model. I had to use different data augmentation methods and preprocess the data to get a slightly even distribution.  

I would like to revisit the project in the future. Below are some areas I would like to tackle:  

* Instead of using Udacity data, collect more data from both tracks. My current result only works for the track 1 and old version of track 2. But not the new version of track 2.
* Use an analog joystick to collect the data. As some people pointed online, with keyboard input, it's not even possible to train it. It's like a case of garbage in and garbage out.
* Use a GPU to train the model. It's indeed frustrating for me to wait hours for the results on CPU.
* It would be a good idea to consider more model outputs other than steering angle. It would be helpful for the model to train throttle and brake and use speed as input as well.

I enjoy the project and am happy to see the car successfully drive autonomously around the track without leaving the road after two weeks work. I hope I would get a chance to work on a real autonomous driving vehicle in the future.
