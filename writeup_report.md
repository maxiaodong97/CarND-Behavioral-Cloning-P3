# **Behavioral Cloning**


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model use modified version of [nVidia CNN model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). (model.py line 127-171)

Following is modified:
1. Remove 1164 fully connected layer to increase the training speed.
2. add cropping layer to crop the "non-road" noise.
3. add dropout layer to prevent overfitting


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 127-171).
The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 199-210)
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an "adam" optimizer, so the learning rate was not tuned manually (model.py line 203).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I drive in the simulator around 20 mph, so I also modified the
drive.py to increase the the speed from 9 to 19 mph.

I used a combination of center lane driving, recovering from the left and right sides of the road, in order to teach the model to
recover from edge position. For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use nVidia CNN model as a start point, repeating with
1. load weights from previous training result.
2. Fit the model with new training data.
3. Save the model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set.
This implied that the model was overfitting. To combat the overfitting, I modified the model by introducing multiple dropout layers.

Then I found both training and validation loss are decreasing after each epochs. I also increase the epochs as long as I found
it keep decreasing.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where
the vehicle fell off the track, to improve the driving behavior in these cases, I create new training data set that focus more
the weak spots, especially recover from left right to center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 127) consisted of a convolution neural network with the following layers and layer sizes


Here is the summary of model:
```python
___________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_3 (Lambda)                (None, 160, 320, 3)   0           lambda_input_3[0][0]             
____________________________________________________________________________________________________
cropping2d_3 (Cropping2D)        (None, 65, 310, 3)    0           lambda_3[0][0]                   
____________________________________________________________________________________________________
convolution2d_11 (Convolution2D) (None, 31, 153, 24)   1824        cropping2d_3[0][0]               
____________________________________________________________________________________________________
dropout_17 (Dropout)             (None, 31, 153, 24)   0           convolution2d_11[0][0]           
____________________________________________________________________________________________________
convolution2d_12 (Convolution2D) (None, 14, 75, 36)    21636       dropout_17[0][0]                 
____________________________________________________________________________________________________
dropout_18 (Dropout)             (None, 14, 75, 36)    0           convolution2d_12[0][0]           
____________________________________________________________________________________________________
convolution2d_13 (Convolution2D) (None, 5, 36, 48)     43248       dropout_18[0][0]                 
____________________________________________________________________________________________________
dropout_19 (Dropout)             (None, 5, 36, 48)     0           convolution2d_13[0][0]           
____________________________________________________________________________________________________
convolution2d_14 (Convolution2D) (None, 3, 34, 64)     27712       dropout_19[0][0]                 
____________________________________________________________________________________________________
dropout_20 (Dropout)             (None, 3, 34, 64)     0           convolution2d_14[0][0]           
____________________________________________________________________________________________________
convolution2d_15 (Convolution2D) (None, 1, 32, 64)     36928       dropout_20[0][0]                 
____________________________________________________________________________________________________
dropout_21 (Dropout)             (None, 1, 32, 64)     0           convolution2d_15[0][0]           
____________________________________________________________________________________________________
flatten_3 (Flatten)              (None, 2048)          0           dropout_21[0][0]                 
____________________________________________________________________________________________________
dense_9 (Dense)                  (None, 100)           204900      flatten_3[0][0]                  
____________________________________________________________________________________________________
dropout_22 (Dropout)             (None, 100)           0           dense_9[0][0]                    
____________________________________________________________________________________________________
dense_10 (Dense)                 (None, 50)            5050        dropout_22[0][0]                 
____________________________________________________________________________________________________
dropout_23 (Dropout)             (None, 50)            0           dense_10[0][0]                   
____________________________________________________________________________________________________
dense_11 (Dense)                 (None, 10)            510         dropout_23[0][0]                 
____________________________________________________________________________________________________
dropout_24 (Dropout)             (None, 10)            0           dense_11[0][0]                   
____________________________________________________________________________________________________
dense_12 (Dense)                 (None, 1)             11          dropout_24[0][0]                 
====================================================================================================
Total params: 341,819
Trainable params: 341,819
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text](https://github.com/maxiaodong97/CarND-Behavioral-Cloning-P3/blob/master/images/center_lane_driving.jpg "center lane driving")

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text](https://github.com/maxiaodong97/CarND-Behavioral-Cloning-P3/blob/master/images/recover_left.jpg "recover left")
![alt text](https://github.com/maxiaodong97/CarND-Behavioral-Cloning-P3/blob/master/images/recover_right.jpg "recover right")

Then I repeated this process on track two in order to get more data points.
To augment the data sat, I also flipped images and angles thinking that this would eliminate the bias that in track one, we always circle anti-clockwise.
For example, here is an image that has then been flipped:





After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
