# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./examples/example_per_class.png "one image per class"
[image1]: ./examples/nr_samples_per_class_initial.jpg "Samples per class"
[image2]: ./examples/original_and_augmented_lighter.png "original and augmented image"
[image3]: ./examples/orig_and_gray.png "original and grayscale image"
[image4]: ./examples/five_new_images_from_internet.png "traffic signs from internet.png"
[image5]: ./examples/prob_per_internet_image_0.png "Traffic Sign 1"
[image6]: ./examples/prob_per_internet_image_1.png "Traffic Sign 2"
[image7]: ./examples/prob_per_internet_image_2.png "Traffic Sign 3"
[image8]: ./examples/prob_per_internet_image_3.png "Traffic Sign 4"
[image9]: ./examples/prob_per_internet_image_4.png "Traffic Sign 5"

---
### Writeup / README

This is my writeup file for the project.
Here is a link to my [project code](https://github.com/Patright/traffic_light_classification/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

After importing the python libraries I loaded the data in three data sets. One for training, one for validation an one for the final testing. Then I used the numpy library to calculate summary statistics of the traffic signs data set:

* Number of training examples   = 34799 => 67%
* Number of validation examples = 4410  =>  9%
* Number of testing examples    = 12630 => 24%
* Image data shape (RGB)        = (32, 32, 3)
* Number of unique classes      = 43

Overview of traffic signs with name:

![alt text][image0]

Traffic sign classes with description.

| ClassID    | Description|  ClassID    | Description| 
|:-----------|:-----------|:-----------|:-----------| 
|  0  | Speed limit (20km/h)|  22 | Bumpy road|
|  1  | Speed limit (30km/h)|  23 | Slippery road|
|  2  | Speed limit (50km/h)|  24 | Road narrows on the right|
|  3  | Speed limit (60km/h)|  25 | Road work|
|  4  | Speed limit (70km/h)|  26 | Traffic signals|
|  5  | Speed limit (80km/h)|  27 | Pedestrians|
|  6  | End of speed limit (80km/h)|  28 | Children crossing|
|  7  | Speed limit (100km/h)|  29 | Bicycles crossing|
|  8  | Speed limit (120km/h)|  30 | Beware of ice/snow|
|  9  | No passing|  31 | Wild animals crossing|
| 10  | No passing for vehicles over 3.5 metric tons|  32 | End of all speed and passing limits|
| 11  | Right-of-way at the next intersection|  33 | Turn right ahead|
| 12  | Priority road|  34 | Turn left ahead|
| 13  | Yield|  35 | Ahead only|
| 14  | Stop|  36 | Go straight or right|
| 15  | No vehicles|  37 | Go straight or left|
| 16  | Vehicles over 3.5 metric tons prohibited|  38 | Keep right|
| 17  | No entry|  39 | Keep left|
| 18  | General caution|  40 | Roundabout mandatory|
| 19  | Dangerous curve to the left|  41 | End of no passing|
|  20 | Dangerous curve to the right|  42 | End of no passing by vehicles over 3.5 metric tons|
|  21 | Double curve|

This bar chart shows the distribution of samples per class for the training data set.
It can clearly be seen that the samples are not uniformly distributed, for there are classes from 180 sample images per class up to 2010 sample images per class.

![alt text][image1]




### Design and Test a Model Architecture

#### 1. Data augmentation: random rotation and random noise

I used random rotation and random noise on the training data set, at first to fill up the classes with less than 990 image samples and than on all classes to increase the overall amount of training data from 34799 to 231831. As result I got a more accurate model, especially for distinguishing between very similar traffic signs like "Slippery road", "Road work", "Children crossing" and "Bicycles crossing". Finally the model was able to classify a new "Slippery road" image which I downloaded from the internet.

Here is an example of a rotated image and a noisy image:

![alt text][image2]

#### 2. Preprocessing the image data.

For preprocessing the images I at first converted them to grayscale and normalized them in a second step.

For conversion to grayscale color space I used the numpy library to calculate the dot product of the R-,G- and B-channels with the factors: 0.2989, 0.5870, 0.1140. 
Than I normalized the grayscale images, by calculating the global mean and the standard deviation, than I subtracted the mean from every pixel value and divided it by the standard deviation, which resulted in a 
mean of the training set of 0.0 and its standard deviation of 1.0 with the minimal value: -1.242 and the maximal value: 2.704.
I also tried to clip the pixel values to a range between minus one and one and than shift them to the positive range from zero to one, but that resulted in a significantly less accurate model.

Here is an example of an RGB-image and the corresponding preprocessed grayscale image:

![alt text][image3]



#### 3. Model architecture of the CNN

I used the "LeNet5"-CNN invented by Yann LeCun [wikipedia link](https://en.wikipedia.org/wiki/LeNet) and modified it by adding dropout layers between the fully connected layers with a dropout rate of 50%. By doing that the model is forced to learn redundant information for classifying the images correctly, that leads to a more robust and accurate model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:----------------------|:----------------------------------------------| 
| Input         		| 32x32x1 normalized grayscale image			| 
| Convolution 3x3     	| 1x1 stride, valid padding, output: 28x28x6 	|
| RELU					| activation function							|
| Max pooling	      	| 2x2 stride, valid padding, output: 14x14x6	|
| Convolution 3x3     	| 1x1 stride, valid padding, output: 10x10x16 	|
| RELU					| activation function							|
| Max pooling	      	| 2x2 stride, valid padding, output: 5x5x16 	|
| Flatten to 1D 		| output: 400									|
| Dropout               | keep probability 0.5                          |
| Fully connected		| output: 120 									|
| Dropout               | keep probability 0.5                          |
| Fully connected		| output: 84 									|
| Dropout               | keep probability 0.5                          |
| Fully connected		| output = logits: 43							|
|        				|                                    			|
| Softmax				| calculate probabilities for logits  			|
| cross entropy 		| calculate distances to correct classes		|
| loss operation		| calculate average cross-entropy    			|
| optimizer         	| use of Adam optimizer							|
| training operation	| minimizing the loss function with gradient descent|
 


#### 3. Parameter for training the model.

In order to train the model, I varied the number of epochs, the batch size, the learning rate and the parameters used for the data augmentation: degree for rotation and mean and variance for noise.

I finally came up with the following values:  
for rotation: -15° to 15°  
for noise the mean between -8 to 8 and the variance between 0 to 100  
number of epochs: 30  
batch size: 128  
learning rate: 0.0005  


#### 4. Results and discussion of the results

My final model results were:
* training set accuracy of 0.98
* validation set accuracy of 0.96
* test set accuracy of 0.94

I first started with the LeNet5-CNN but the results were not good enough, the accuracy for the validation data set was less than 0.93. As a next step I decided to add dropout layers between the three fully connected layers. As widely recommended I chose 0.5 for the keep_prob parameter. With that optimization the validation set accuracy reached the goal of 0.93. But when using the model on my five randomly chosen traffic signs I found on the web, I got a poor result for the "slippery road" sign. The reason is that there are some very similar traffic signs all with a red framed triangle on a white background and a black drawing in it. Besides their similarity, the amount of training data especially for that kind of traffic signs was relatively low. To address that issue I decided to take a step back and add augmented data to the training data set by randomly rotating and adding of noise to the images. And finally, I got a good detection probability for the "Slippery road" sign.



### Test the model on new images

#### 1. Five German traffic signs found on the web

Here are the five German traffic signs that I found on the web by chance:

![alt text][image4]

I cropped and resized them in a way, that they fit as input for my model.
The first three images should be relatively easy to identify, because of their clear characteristics, and the huge amount of sample images in the training data set.
Image four "Wild animals crossing" and image five "Slippery road" might be more difficult to classify because they both have the red framed triangle shape on a white background and a black drawing in it.

#### 2. Discussion of the model's predictions on the new traffic signs.

As estimated the first three images got a prediction probability > 0.98 right away.
Surprisingly the "Wild animals crossing" sign also got very good prediction probability of > 0.98, whereas the "Slipper road" sign got the previously expected very low prediction probability. After optimizing the model as described above, I got a prediction probability > 0.55 (in some cases with e.g. more training epochs the prediction went up to 85%).



Here are the results of the prediction:

| Image			        |     Prediction	   | 
|:----------------------|:---------------------| 
| Speed limit (30km/h)  | Speed limit (30km/h) | 
| no passing   			| no passing           |
| Yield					| Yield                |
| wild animals crossing | wild animals crossing|
| Slippery Road			| Slippery Road        |


The model was able to correctly guess all five traffic signs, which gives an accuracy of 100%. This is even better than the accuracy of the test data set, but there are of course other influences on the detection rate for example weather, brightness, viewpoint, cleanliness of the traffic sign, etc. that influence the accuracy. In summary I do think that the prediction results for the five traffic signs found on the web fit very well with the test set accuracy.

#### 3. Description of the model's predictions on the new traffic signs.

At first I calculated the logits for every new image, by feeding the images with their corresponding labels into the model. Then I used the tf.nn.top_k() with k=5 on the tf.nn.softmax() of the logits in order to get the top five probabilities for every image (I did this in a tf.Session() because I used the Tensorflow 1 API, which does not support eager execution).

For the first image, the model is nearly a hundred percent sure that this is a "Speed limit (30km/h)" sign (probability of 1.0), the top five soft max probabilities are:

![alt text][image5]

For the second image the probability for the "no passing" sign is 0.98, followed by 2% probability for "End of no passing":

![alt text][image6]

For the third and the forth image I got a 100% probability for the correct detection of "Yield" and "Wild animals crossing":

![alt text][image7]

![alt text][image8]

The fifth image is the most challenging. The model predicted with the highest probability the correct sign (0.55 for "Slippery road"), with a probability of 0.11 "Beware of ice/snow" with 0.10 "Right-of-way at the next intersection" with 0.04 "End of no passing by vehicles over 3.5 metric tons" and with the lowest of the top five softmax probabilities of 0.04 for the "Priority road" sign.

![alt text][image9]
