# **Behavioral Cloning** 

## Writeup

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
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64.

The model includes PReLU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model showed no sign of overfitting so no action was done regarding that.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the NVIDIA architecture and build from there.

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because they use this architecture for similar problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

The results while training were good both for the training and validation sets, but when I run the simulator, the car fail to stay on track.

To combat this, I modified the data stracture so it will be more stratified.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
- lambda layer.
- 4 CNN with 3x3, 3x3, 3x3, 5x5 filters, respectively.
- followd by PReLU and BatchNormalization layers (after each of the CNN layers).
- 3 fully connected layers (100, 50, 10).
- output layer.

#### 3. Creation of the Training Set & Training Process

I used the udacity dataset as my starting point.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in case he will drifted aside.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would provide the model with more data to work with.

After the collection process, I had 53,058 data points. I then preprocessed this data by stratifying it, so I left with 13014. inside the generator I turn the data into gray scale.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by lowest score the validation set got. I used an adam optimizer so that manually training the learning rate wasn't necessary.