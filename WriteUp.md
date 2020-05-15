# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Training.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_Enhanced.h5 containing a trained convolution neural network 
* WriteUp.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_Enhanced.h5
```

#### 3. Submission code is usable and readable

The Training.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the *Nvidia End-to-End Deep Learning for Self-Driving Cars*. This model consists of:

* One Normalizing keras lambda layer (Training.py line 24)
* Three 5x5 Convolutional layers, with 2x2 strides (Training.py lines 25 - 30)
* Two 3x3 Convolutional layers, with 1x1 strides (Training.py lines 31 - 34)
* A flattening layer (Training.py line 35)
* Four fully connected layers (ouput 100, 50, 10, 1) (Training.py lines 37 - 43)

This model includes RELU layers to introduce nonlinearity.

Since the model has made its proof as we can read [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), it provides us a good basis.

#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layers in order to reduce overfitting (Training.py line 36). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (Training.py lines 36). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (Training.py line 45).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was mainly to combat overfitting.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had low mean squared errors on both the training and validation sets, but still they were mot matching. Also after 5 epochs, errors on the validation started raising again. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that both mean square errors on validation and training sets were low and quite same, and I reduced the number of epochs to 5.

The final step was to run the simulator to see how well the car was driving around track one. There vehicle had it hard to adjust in turns. To improve the driving behavior in these cases, I added more images of turning cases to help the model adjusting in these cases. Also to help the neural network focusing on the driving , I added a cropping layer that focuses only on the road patterns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (Training.py lines 22-43) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 images   					        | 
| Cropping(50,20), (0,0)| 90x320x3 images   					        | 
| Normalization         | 90x320x3 images   					        | 
| Convolution 5x5    	| 2x2 stride, valid padding, outputs 43x158x24 	|
| RELU					|												|
| Convolution 5x5    	| 2x2 stride, valid padding, outputs 20x77x36 	|
| RELU					|												|
| Convolution 5x5    	| 2x2 stride, valid padding, outputs 8x37x48 	|
| RELU					|												|
| Convolution 3x3    	| 1x1 stride, valid padding, outputs 6x35x64 	|
| RELU					|												|
| Convolution 3x3    	| 1x1 stride, valid padding, outputs 4x33x64 	|
| RELU					|												|
| Flatten               |                                   			|
| Dropout           	| Keep probability 0.5 				            |
| Fully connection 1500	| outputs 100  									|
| RELU					|												|
| Fully connection 100	| outputs 50  									|
| RELU					|												|
| Fully connection 50	| outputs 10  									|
| RELU					|												|
| Fully connection 10	| output 1  									|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

<img src="WriteUp_Images\Center_Example.jpg" width="800"/>

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center of the lane. These images show what a recovery looks like starting from the right side of the lane steering back to the center of the lane:

|  |   | | |
|:---:|:---:|:---:|:---:|
| <img src="WriteUp_Images\Center_Recovery_1.jpg" width="100"/> | <img src="WriteUp_Images\Center_Recovery_2.jpg" width="100"/> | <img src="WriteUp_Images\Center_Recovery_3.jpg" width="100"/> | <img src="WriteUp_Images\Center_Recovery_4.jpg" width="100"/> |<img src="WriteUp_Images\Center_Recovery_5.jpg" width="100"/> |
|<img src="WriteUp_Images\Center_Recovery_6.jpg" width="100"/> | <img src="WriteUp_Images\Center_Recovery_7.jpg" width="100"/> | <img src="WriteUp_Images\Center_Recovery_8.jpg" width="100"/> | <img src="WriteUp_Images\Center_Recovery_9.jpg" width="100"/> | <img src="WriteUp_Images\Center_Recovery_10.jpg" width="100"/>|

To augment the data set, I also flipped images and angles thinking that this would simulate a right turn using a left turn and vice versa. The steering angle for the flipped image correspond to the opposite of the steering angle of the original image. For example, here is an image that has then been flipped:

<img src="WriteUp_Images\Flip.jpg" width="1000"/>

Another data augmentation consisted of using the images of the right and left cameras. The steering angle for those images were derived by correcting the one for the center camera:
```sh
left steering angle = center steering angle + correction
right steering angle = center steering angle - correction
```
An example of images from the different cameras at the same position:

|Left camera  | Center Camera| Right Camera  |
|:---:|:---:|:---:|
| <img src="WriteUp_Images\Camera_Left.jpg" width="200"/> | <img src="WriteUp_Images\Camera_Center.jpg" width="200"/> | <img src="WriteUp_Images\Camera_Right.jpg" width="200"/> | 

After the collection process, I had 78786 number of data points. I then preprocessed this data by adding the cropping and normalization layers to the model architecture.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced because the models used to start overfitting after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
